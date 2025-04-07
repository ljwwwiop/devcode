import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union, cast
import ipdb
import torchvision.models
from typing_extensions import Literal
from util.set_distance import chamfer, hausdorff
# from model import pointnet_cls
# from model.feature_models import FeatureModel
from pytorch3d.ops import knn_points, knn_gather


class L2Dist(nn.Module):

    def __init__(self):
        """Compute global L2 distance between two point clouds.
        """
        super(L2Dist, self).__init__()

    def forward(self, adv_pc, ori_pc,
                weights=None, batch_avg=True):
        """Compute L2 distance between two point clouds.
        Apply different weights for batch input for CW attack.

        Args:
            adv_pc (torch.FloatTensor): [B, K, 3] or [B, 3, K]
            ori_pc (torch.FloatTensor): [B, K, 3] or [B, 3, k]
            weights (torch.FloatTensor, optional): [B], if None, just use avg
            batch_avg: (bool, optional): whether to avg over batch dim
        """
        B = adv_pc.shape[0]
        if weights is None:
            weights = torch.ones((B,))
        weights = weights.float().cuda()
        dist = torch.sqrt(torch.sum((adv_pc - ori_pc) ** 2, dim=[1, 2]) + torch.tensor(1e-7))  # [B]
        dist = dist * weights
        if batch_avg:
            return dist.mean()
        return dist


class ChamferDist(nn.Module):

    def __init__(self, method='adv2ori'):
        """Compute chamfer distance between two point clouds.

        Args:
            method (str, optional): type of chamfer. Defaults to 'adv2ori'.
        """
        super(ChamferDist, self).__init__()

        self.method = method

    def forward(self, adv_pc, ori_pc,
                weights=None, batch_avg=True):
        """Compute chamfer distance between two point clouds.

        Args:
            adv_pc (torch.FloatTensor): [B, K, 3]
            ori_pc (torch.FloatTensor): [B, K, 3]
            weights (torch.FloatTensor, optional): [B], if None, just use avg
            batch_avg: (bool, optional): whether to avg over batch dim
        """

        if adv_pc.shape[1] == 3:
            adv_pc = adv_pc.permute(0,2,1)
        if ori_pc.shape[1] == 3:
            ori_pc = ori_pc.permute(0,2,1)
        # import pdb; pdb.set_trace()
        B = adv_pc.shape[0]
        if weights is None:
            weights = torch.ones((B,)).to(adv_pc.device)
        loss1, loss2 = chamfer(adv_pc, ori_pc)  # [B], adv2ori, ori2adv
        if self.method == 'adv2ori':
            loss = loss1
        elif self.method == 'ori2adv':
            loss = loss2
        else:
            loss = (loss1 + loss2) / 2.
        weights = weights.float().cuda()
        loss = loss * weights
        if batch_avg:
            return loss.mean()
        return loss


class HausdorffDist(nn.Module):

    def __init__(self, method='adv2ori'):
        """Compute hausdorff distance between two point clouds.

        Args:
            method (str, optional): type of hausdorff. Defaults to 'adv2ori'.
        """
        super(HausdorffDist, self).__init__()

        self.method = method

    def forward(self, adv_pc, ori_pc,
                weights=None, batch_avg=True):
        """Compute hausdorff distance between two point clouds.

        Args:
            adv_pc (torch.FloatTensor): [B, K, 3]
            ori_pc (torch.FloatTensor): [B, K, 3]
            weights (torch.FloatTensor, optional): [B], if None, just use avg
            batch_avg: (bool, optional): whether to avg over batch dim
        """
        B = adv_pc.shape[0]

        if adv_pc.shape[1] == 3:
            adv_pc = adv_pc.permute(0,2,1)
        if ori_pc.shape[1] == 3:
            ori_pc = ori_pc.permute(0,2,1)

        if weights is None:
            weights = torch.ones((B,))
        loss1, loss2 = hausdorff(adv_pc, ori_pc)  # [B], adv2ori, ori2adv
        if self.method == 'adv2ori':
            loss = loss1
        elif self.method == 'ori2adv':
            loss = loss2
        else:
            loss = (loss1 + loss2) / 2.
        weights = weights.float().cuda()
        loss = loss * weights
        if batch_avg:
            return loss.mean()
        return loss


class KNNDist(nn.Module):

    def __init__(self, k=5, alpha=1.05):
        """Compute kNN distance punishment within a point cloud.

        Args:
            k (int, optional): kNN neighbor num. Defaults to 5.
            alpha (float, optional): threshold = mean + alpha * std. Defaults to 1.05.
        """
        super(KNNDist, self).__init__()

        self.k = k
        self.alpha = alpha

    def forward(self, pc, weights=None, batch_avg=True):
        """KNN distance loss described in AAAI'20 paper.

        Args:
            adv_pc (torch.FloatTensor): [B, K, 3]
            weights (torch.FloatTensor, optional): [B]. Defaults to None.
            batch_avg: (bool, optional): whether to avg over batch dim
        """
        # build kNN graph
        B, K = pc.shape[:2]
        if pc.shape[1] != 3:
            pc = pc.transpose(2, 1)  # [B, 3, K]
        inner = -2. * torch.matmul(pc.transpose(2, 1), pc)  # [B, K, K]
        xx = torch.sum(pc ** 2, dim=1, keepdim=True)  # [B, 1, K]
        dist = xx + inner + xx.transpose(2, 1)  # [B, K, K], l2^2
        # print(dist.min().item())

        # assert dist.min().item() >= -1e-6

        # the min is self so we take top (k + 1)
        neg_value, _ = (-dist).topk(k=self.k + 1, dim=-1)
        # [B, K, k + 1]
        value = -(neg_value[..., 1:])  # [B, K, k]
        value = torch.mean(value, dim=-1)  # d_p, [B, K]
        with torch.no_grad():
            mean = torch.mean(value, dim=-1)  # [B]
            std = torch.std(value, dim=-1)  # [B]
            # [B], penalty threshold for batch
            threshold = mean + self.alpha * std
            weight_mask = (value > threshold[:, None]). \
                float().detach()  # [B, K]
        loss = torch.mean(value * weight_mask, dim=1)  # [B]
        # accumulate loss
        if weights is None:
            weights = torch.ones((B,))
        weights = weights.float().cuda()
        loss = loss * weights
        if batch_avg:
            return loss.mean()
        return loss


class LaplacianDist(nn.Module):

    def __init__(self, k):
        """Compute global L2 distance between two point clouds.
        """
        super(LaplacianDist, self).__init__()
        self.k = k

    def forward(self, adv_pc, ori_pc, nearest_indices,
                weights=None, batch_avg=True):
        """Compute L2 distance between two point clouds.
        Apply different weights for batch input for CW attack.

        Args:
            adv_pc (torch.FloatTensor): [B, K, 3] or [B, 3, K]
            ori_pc (torch.FloatTensor): [B, K, 3] or [B, 3, k]
            nearest_indices (torch.Tensor): [B, K, k]
            weights (torch.FloatTensor, optional): [B], if None, just use avg
            batch_avg: (bool, optional): whether to avg over batch dim
        """
        B = adv_pc.shape[0]
        if weights is None:
            weights = torch.ones((B,))
        weights = weights.float().cuda()

        delta = adv_pc - ori_pc  # [B, 3, K]
        # (delta_vi - delta_vq)  shape is  [B, K, k]
        # 输出和index的shape是一样的，delta现在是每个点的变化量，目标是每个点的邻近k个点的变化量，首先要求[B,3,K,k]
        delta = delta.unsqueeze(3).expand(-1, -1, -1, nearest_indices.shape[2])
        # need check
        neighboring_points = torch.gather(delta, 2, nearest_indices.unsqueeze(1).expand(-1, 3, -1, -1))  # [B, 3, K, k]

        norm_squared = torch.norm(neighboring_points, dim=1) ** 2  # [B, K, k]
        dist = torch.sum(norm_squared, dim=[1, 2])  # [B]
        dist = dist * weights
        if batch_avg:
            return dist.mean()
        return dist

    def KNN_indices(self, x):
        pc = x.clone().detach().double()  # [B, 3, K]
        B, _, K = pc.shape
        #        pc = pc.transpose(2, 1)
        inner = -2. * torch.matmul(pc.transpose(2, 1), pc)  # [B, K, K]
        xx = torch.sum(pc ** 2, dim=1, keepdim=True)  # [B, 1, K]
        dist = xx + inner + xx.transpose(2, 1)  # [B, K, K]
        assert dist.min().item() >= -1e-6
        # the min is self so we take top (k + 1)
        neg_value, nearest_indices = (-dist).topk(k=self.k + 1, dim=-1)  # [B, K, k + 1]
        value = -(neg_value[..., 1:])  # [B, K, k] 返回对于全部K个点，与相邻最短的k个点之间的距离
        nearest_indices = nearest_indices[..., 1:]  # [B, K, k]
        return value, nearest_indices


# class LaplacianDist(nn.Module):
#
#     def __init__(self, k):
#         """Compute global L2 distance between two point clouds.
#         """
#         super(LaplacianDist, self).__init__()
#         self.k = k
#
#     def forward(self, pc, batch_avg=True):
#         """Compute Laplacian loss of Point Cloud
#         Args:
#             ori_pc (torch.FloatTensor): [B, K, 3] or [B, 3, k]
#             batch_avg: (bool, optional): whether to avg over batch dim
#         """
#         inter_KNN = knn_points(pc.permute(0, 2, 1),
#                                pc.permute(0, 2, 1), K=self.k + 1)  # [dists:[b,n,k+1], idx:[b,n,k+1]]
#         nn_pts = (knn_gather(pc.permute(0, 2, 1), inter_KNN.idx).permute(0, 3, 1, 2)[:, :, :, 1:]
#                   .contiguous())  # [b, 3, n ,k]
#         delta = pc - torch.sum(nn_pts / self.k, dim=3)
#         if batch_avg == True:
#             loss = torch.sum(torch.norm(delta) ** 2)
#         else:
#             loss = torch.norm(delta, dim=(1, 2))
#         return loss


class ChamferkNNDist(nn.Module):

    def __init__(self, chamfer_method='adv2ori',
                 knn_k=5, knn_alpha=1.05,
                 chamfer_weight=5., knn_weight=3.):
        """Geometry-aware distance function of AAAI'20 paper.

        Args:
            chamfer_method (str, optional): chamfer. Defaults to 'adv2ori'.
            knn_k (int, optional): k in kNN. Defaults to 5.
            knn_alpha (float, optional): alpha in kNN. Defaults to 1.1.
            chamfer_weight (float, optional): weight factor. Defaults to 5..
            knn_weight (float, optional): weight factor. Defaults to 3..
        """
        super(ChamferkNNDist, self).__init__()

        self.chamfer_dist = ChamferDist(method=chamfer_method)
        self.knn_dist = KNNDist(k=knn_k, alpha=knn_alpha)
        self.w1 = chamfer_weight
        self.w2 = knn_weight

    def forward(self, adv_pc, ori_pc,
                weights=None, batch_avg=True):
        """Adversarial constraint function of AAAI'20 paper.

        Args:
            adv_pc (torch.FloatTensor): [B, K, 3]
            ori_pc (torch.FloatTensor): [B, K, 3]
            weights (np.array): weight factors
            batch_avg: (bool, optional): whether to avg over batch dim
        """
        chamfer_loss = self.chamfer_dist(
            adv_pc, ori_pc, weights=weights, batch_avg=batch_avg)
        knn_loss = self.knn_dist(
            adv_pc, weights=weights, batch_avg=batch_avg)
        loss = chamfer_loss * self.w1 + knn_loss * self.w2
        return loss


class FarthestDist(nn.Module):

    def __init__(self):
        """Used in adding cluster attack.
        """
        super(FarthestDist, self).__init__()

    def forward(self, adv_pc, weights=None, batch_avg=True):
        """Compute the farthest pairwise point dist in each added cluster.

        Args:
            adv_pc (torch.FloatTensor): [B, num_add, cl_num_p, 3]
            weights (np.array): weight factors
            batch_avg: (bool, optional): whether to avg over batch dim
        """
        B = adv_pc.shape[0]
        if weights is None:
            weights = torch.ones((B,))
        delta_matrix = adv_pc[:, :, None, :, :] - adv_pc[:, :, :, None, :] + 1e-7
        # [B, num_add, num_p, num_p, 3]
        norm_matrix = torch.norm(delta_matrix, p=2, dim=-1)  # [B, na, np, np]
        max_matrix = torch.max(norm_matrix, dim=2)[0]  # take the values of max
        far_dist = torch.max(max_matrix, dim=2)[0]  # [B, num_add]
        far_dist = torch.sum(far_dist, dim=1)  # [B]
        weights = weights.float().cuda()
        loss = far_dist * weights
        if batch_avg:
            return loss.mean()
        return loss


class FarChamferDist(nn.Module):

    def __init__(self, num_add, chamfer_method='adv2ori',
                 chamfer_weight=0.1):
        """Distance function used in generating adv clusters.
        Consisting of a Farthest dist and a chamfer dist.

        Args:
            num_add (int): number of added clusters.
            chamfer_method (str, optional): chamfer. Defaults to 'adv2ori'.
            chamfer_weight (float, optional): weight factor. Defaults to 0.1.
        """
        super(FarChamferDist, self).__init__()

        self.num_add = num_add
        self.far_dist = FarthestDist()
        self.chamfer_dist = ChamferDist(method=chamfer_method)
        self.cd_w = chamfer_weight

    def forward(self, adv_pc, ori_pc,
                weights=None, batch_avg=True):
        """Adversarial constraint function of CVPR'19 paper for adv clusters.

        Args:
            adv_pc (torch.FloatTensor): [B, num_add * cl_num_p, 3],
                                        the added clusters
            ori_pc (torch.FloatTensor): [B, K, 3]
            weights (np.array): weight factors
            batch_avg: (bool, optional): whether to avg over batch dim
        """
        B = adv_pc.shape[0]
        chamfer_loss = self.chamfer_dist(
            adv_pc, ori_pc, weights=weights, batch_avg=batch_avg)
        adv_clusters = adv_pc.view(B, self.num_add, -1, 3)
        far_loss = self.far_dist(
            adv_clusters, weights=weights, batch_avg=batch_avg)
        loss = far_loss + chamfer_loss * self.cd_w
        return loss


class L2ChamferDist(nn.Module):

    def __init__(self, num_add, chamfer_method='adv2ori',
                 chamfer_weight=0.2):
        """Distance function used in generating adv objects.
        Consisting of a L2 dist and a chamfer dist.

        Args:
            num_add (int): number of added objects.
            chamfer_method (str, optional): chamfer. Defaults to 'adv2ori'.
            chamfer_weight (float, optional): weight factor. Defaults to 0.2.
        """
        super(L2ChamferDist, self).__init__()

        self.num_add = num_add
        self.chamfer_dist = ChamferDist(method=chamfer_method)
        self.cd_w = chamfer_weight
        self.l2_dist = L2Dist()

    def forward(self, adv_pc, ori_pc, adv_obj, ori_obj,
                weights=None, batch_avg=True):
        """Adversarial constraint function of CVPR'19 paper for adv objects.

        Args:
            adv_pc (torch.FloatTensor): [B, num_add * obj_num_p, 3],
                                        the added objects after rot and shift
            ori_pc (torch.FloatTensor): [B, K, 3]
            adv_obj (torch.FloatTensor): [B, num_add, obj_num_p, 3],
                                        the added objects after pert
            ori_pc (torch.FloatTensor): [B, num_add, obj_num_p, 3],
                                        the clean added objects
            weights (np.array): weight factors
            batch_avg: (bool, optional): whether to avg over batch dim
        """
        B = adv_pc.shape[0]
        chamfer_loss = self.chamfer_dist(
            adv_pc, ori_pc, weights=weights, batch_avg=batch_avg)
        l2_loss = self.l2_dist(
            adv_obj.view(B, -1, 3), ori_obj.view(B, -1, 3),
            weights=weights, batch_avg=batch_avg)
        loss = l2_loss + self.cd_w * chamfer_loss
        return loss

class CurvStdDist(nn.Module):
    def __init__(self, k=5):
        super(CurvStdDist, self).__init__()
        self.k = k

    def forward(self, ori_data, adv_data, ori_normal):
        pdist = torch.nn.PairwiseDistance(p=2)
        # fixme adv_data 使用 ori_normal 的影响有多大
        ori_kappa_std = self._get_kappa_std_ori(ori_data, ori_normal, k=self.k)  # [b, n]
        adv_kappa_std = self._get_kappa_std_ori(adv_data, ori_normal, k=self.k)  # [b, n]
        curv_std_dist = pdist(ori_kappa_std, adv_kappa_std).mean()
        return curv_std_dist

    def _get_kappa_std_ori(self, pc, normal, k=10):
        b, _, n = pc.size()
        # inter_dis = ((pc.unsqueeze(3) - pc.unsqueeze(2))**2).sum(1)
        # inter_idx = torch.topk(inter_dis, k+1, dim=2, largest=False, sorted=True)[1][:, :, 1:].contiguous()
        # nn_pts = torch.gather(pc, 2, inter_idx.view(b,1,n*k).expand(b,3,n*k)).view(b,3,n,k)
        inter_KNN = knn_points(pc.permute(0, 2, 1), pc.permute(0, 2, 1), K=k + 1)  # [dists:[b,n,k+1], idx:[b,n,k+1]]
        nn_pts = knn_gather(pc.permute(0, 2, 1), inter_KNN.idx).permute(0, 3, 1, 2)[:, :, :,
                 1:].contiguous()  # [b, 3, n ,k]
        vectors = nn_pts - pc.unsqueeze(3)
        vectors = self._normalize(vectors)

        kappa_ori = torch.abs((vectors * normal.unsqueeze(3)).sum(1)).mean(2)  # [b, n]
        nn_kappa = knn_gather(kappa_ori.unsqueeze(2), inter_KNN.idx).permute(0, 3, 1, 2)[:, :, :,
                   1:].contiguous()  # [b, 1, n ,k]
        std_kappa = torch.std(nn_kappa.squeeze(1), dim=2)
        return std_kappa

    def _normalize(self, input, p=2, dim=1, eps=1e-12):
        return input / input.norm(p, dim, keepdim=True).clamp(min=eps).expand_as(input)


class CurvDist(nn.Module):
    def __init__(self, curv_loss_knn=2):
        super(CurvDist, self).__init__()
        self.curv_loss_knn = curv_loss_knn

    def forward(self, ori_data, adv_data, ori_normal):
        ori_kappa = self._get_kappa_ori(ori_data, ori_normal)  # [b, n]
        adv_kappa, normal_curr_iter = self._get_kappa_adv(adv_data, ori_data, ori_normal,
                                                          self.curv_loss_knn)
        curv_loss = self.curvature_loss(adv_data, ori_data, adv_kappa, ori_kappa)
        return curv_loss.mean()

    def _get_kappa_ori(self, pc, normal, k=2):
        b, _, n = pc.size()
        # inter_dis = ((pc.unsqueeze(3) - pc.unsqueeze(2))**2).sum(1)
        # inter_idx = torch.topk(inter_dis, k+1, dim=2, largest=False, sorted=True)[1][:, :, 1:].contiguous()
        # nn_pts = torch.gather(pc, 2, inter_idx.view(b,1,n*k).expand(b,3,n*k)).view(b,3,n,k)
        inter_KNN = knn_points(pc.permute(0, 2, 1), pc.permute(0, 2, 1), K=k + 1)  # [dists:[b,n,k+1], idx:[b,n,k+1]]
        nn_pts = knn_gather(pc.permute(0, 2, 1), inter_KNN.idx).permute(0, 3, 1, 2)[:, :, :,
                 1:].contiguous()  # [b, 3, n ,k]
        vectors = nn_pts - pc.unsqueeze(3)
        vectors = self._normalize(vectors)
        return torch.abs((vectors * normal.unsqueeze(3)).sum(1)).mean(2)  # [b, n]

    def _get_kappa_adv(self, adv_pc, ori_pc, ori_normal, k=2):
        b, _, n = adv_pc.size()
        # compute knn between advPC and oriPC to get normal n_p
        # intra_dis = ((adv_pc.unsqueeze(3) - ori_pc.unsqueeze(2))**2).sum(1)
        # intra_idx = torch.topk(intra_dis, 1, dim=2, largest=False, sorted=True)[1]
        # normal = torch.gather(ori_normal, 2, intra_idx.view(b,1,n).expand(b,3,n))
        intra_KNN = knn_points(adv_pc.permute(0, 2, 1), ori_pc.permute(0, 2, 1), K=1)  # [dists:[b,n,1], idx:[b,n,1]]
        normal = knn_gather(ori_normal.permute(0, 2, 1), intra_KNN.idx).permute(0, 3, 1, 2).squeeze(
            3).contiguous()  # [b, 3, n]

        # compute knn between advPC and itself to get \|q-p\|_2
        # inter_dis = ((adv_pc.unsqueeze(3) - adv_pc.unsqueeze(2))**2).sum(1)
        # inter_idx = torch.topk(inter_dis, k+1, dim=2, largest=False, sorted=True)[1][:, :, 1:].contiguous()
        # nn_pts = torch.gather(adv_pc, 2, inter_idx.view(b,1,n*k).expand(b,3,n*k)).view(b,3,n,k)
        inter_KNN = knn_points(adv_pc.permute(0, 2, 1), adv_pc.permute(0, 2, 1),
                               K=k + 1)  # [dists:[b,n,k+1], idx:[b,n,k+1]]
        nn_pts = knn_gather(adv_pc.permute(0, 2, 1), inter_KNN.idx).permute(0, 3, 1, 2)[:, :, :,
                 1:].contiguous()  # [b, 3, n ,k]
        vectors = nn_pts - adv_pc.unsqueeze(3)
        vectors = self._normalize(vectors)

        return torch.abs((vectors * normal.unsqueeze(3)).sum(1)).mean(2), normal  # [b, n], [b, 3, n]

    def curvature_loss(self, adv_pc, ori_pc, adv_kappa, ori_kappa, k=2):
        b, _, n = adv_pc.size()

        # intra_dis = ((input_curr_iter.unsqueeze(3) - pc_ori.unsqueeze(2))**2).sum(1)
        # intra_idx = torch.topk(intra_dis, 1, dim=2, largest=False, sorted=True)[1]
        # knn_theta_normal = torch.gather(theta_normal, 1, intra_idx.view(b,n).expand(b,n))
        # curv_loss = ((curv_loss - knn_theta_normal)**2).mean(-1)

        intra_KNN = knn_points(adv_pc.permute(0, 2, 1), ori_pc.permute(0, 2, 1), K=1)  # [dists:[b,n,1], idx:[b,n,1]]
        onenn_ori_kappa = torch.gather(ori_kappa, 1, intra_KNN.idx.squeeze(-1)).contiguous()  # [b, n]

        curv_loss = ((adv_kappa - onenn_ori_kappa) ** 2).mean(-1)

        return curv_loss

    def _normalize(self, input, p=2, dim=1, eps=1e-12):
        return input / input.norm(p, dim, keepdim=True).clamp(min=eps).expand_as(input)

## manifold theta loss
class MainFoldTheta(nn.Module):

    def __init__(self, add_type='l1'):
        """Base the manifold geodesic theta (P1,P2) the shortest path
        """
        super(MainFoldTheta, self).__init__()
        self.add = add_type

    def forward(self, adv_pc, ori_pc, key_pc_idx, sm_func, weights=None, batch_avg=True):
        B = adv_pc.shape[0]
        if weights is None:
            weights = torch.ones((B,))
        ori_pc_sm = sm_func(ori_pc) # [b,3,1024]
        adv_pc_sm = sm_func(adv_pc) # []
        # ipdb.set_trace()
        
        ori_pc_sm = ori_pc.permute(0,2,1)
        adv_pc_sm = adv_pc.permute(0,2,1)
        loss = self.parallelism_loss(ori_pc_sm, adv_pc_sm)

        weights = weights.float().cuda()
        loss = loss * weights
        # if batch_avg:
        #     return loss.mean()

        return loss
    
    def point_cloud_manifold_constraint_loss(self, pre_attack_points, post_attack_points):
        # 计算点云的局部协方差矩阵
        def compute_local_covariance_matrix(point_cloud):
            batch_size = point_cloud.shape[0]
            num_points = point_cloud.shape[1]
            covariances = torch.zeros((batch_size, 3, 3))
            for i in range(batch_size):
                points = point_cloud[i]
                centroid = torch.mean(points, dim=0)
                centered_points = points - centroid
                covariance_matrix = torch.mm(centered_points.T, centered_points) / (num_points - 1)
                covariances[i] = covariance_matrix
            return covariances

        # ipdb.set_trace()
        covariance_pre = compute_local_covariance_matrix(pre_attack_points)
        covariance_post = compute_local_covariance_matrix(post_attack_points)

        # 计算协方差矩阵的 Frobenius 范数之差
        frobenius_norm_diff = torch.norm(covariance_pre - covariance_post, 'fro')

        # # 计算点云在球面流形上的测地线距离
        # geodesic_distances_pre = self.compute_geodesic_distances_sphere(pre_attack_points)
        # geodesic_distances_post = self.compute_geodesic_distances_sphere(post_attack_points)

        # # 计算测地线距离的均值之差
        # mean_geodesic_dist_diff = torch.abs(torch.mean(geodesic_distances_pre) - torch.mean(geodesic_distances_post))

        # 综合损失
        # loss = frobenius_norm_diff + mean_geodesic_dist_diff
        loss = frobenius_norm_diff

        return loss

    def compute_geodesic_distances_sphere(self, point_cloud):
        num_points = point_cloud.shape[1]
        B = num_points.shape[0]
        distances = torch.zeros((B, num_points, num_points))
        for i in range(num_points):
            dot_products = torch.dot(point_cloud[:, i], point_cloud[:, i + 1:])
            distances[i, i + 1:] = torch.acos(torch.clamp(dot_products, -1.0, 1.0))
            distances[i + 1:, i] = distances[i, i + 1:]
        return distances

    def calculate_geodesic_angles(self, point_cloud, point_cloud_ori, indices):

        point_cloud = point_cloud.permute(0, 2, 1)
        point_cloud_ori = point_cloud_ori.permute(0,2,1)
        selected_key_points = torch.gather(point_cloud, 1, indices.unsqueeze(-1).expand(-1, -1, 3))
        temp2 = point_cloud.unsqueeze(2)
        dist = torch.norm(temp2 - selected_key_points.unsqueeze(1),dim=-1)

        expanded_indices = indices.unsqueeze(1)
        # dist[torch.arange(indices.shape[0]).unsqueeze(1), expanded_indices] = float('inf')

        # 筛选出距离大于 0 的位置
        valid_distances = (dist > 0.)

        # 在有效距离中找到最小值的索引
        min_values, nearest_indices = torch.min(dist.masked_fill(~valid_distances, float('inf')), dim=1)

        # 获取最近点
        nearest_points = point_cloud[torch.arange(256).unsqueeze(1), nearest_indices]
        # nearest_points = point_cloud_ori[torch.arange(256).unsqueeze(1), nearest_indices]
        # selected_key_points = torch.gather(point_cloud_ori, 1, indices.unsqueeze(-1).expand(-1, -1, 3))
        
        # # 计算夹角
        dot_products = torch.sum((selected_key_points - nearest_points) * nearest_points, dim=-1)
        norms_product = torch.norm(selected_key_points - nearest_points, dim=-1) * torch.norm(nearest_points, dim=-1)
        # 防止分母为 0 导致 NaN
        valid_norms_product = torch.where(norms_product == 0, torch.tensor(1.0).to(norms_product.device), norms_product)

        cos_angles = dot_products / valid_norms_product
        angles = torch.acos(cos_angles)

        # 检查并处理可能的 NaN 值
        angles = torch.nan_to_num(angles)

        # angles = compute_local_covariance_matrix(nearest_points)
        return nearest_points, angles

    def parallelism_loss(self, original_vectors, attacked_vectors):
        '''
            cos(\theta) = <A1,A2>/<||A1||*||A2||>
        '''
        # 计算向量的点积
        dot_products = torch.sum(original_vectors * attacked_vectors, dim=-1)  # 逐元素相乘然后求和
        # 计算向量的模长
        original_norms = torch.norm(original_vectors, dim=2)
        original_norms[original_norms == 0] = 1e-6  # 给零向量的模长赋一个小值
        attacked_norms = torch.norm(attacked_vectors, dim=2)
        attacked_norms[attacked_norms == 0] = 1e-6  # 给零向量的模长赋一个小值
        # 计算余弦相似度
        cos_similarities = dot_products / (original_norms * attacked_norms)
        # 平行度偏差 = 1 - 余弦相似度
        parallelism_deviation = 1 - cos_similarities
        # 损失为平行度偏差的均值
        loss = torch.mean(parallelism_deviation)

        return loss


def normalize_flatten_features(
        features: Tuple[torch.Tensor, ...],
        eps=1e-10,
) -> torch.Tensor:
    """
    Given a tuple of features (layer1, layer2, layer3, ...) from a network,
    flattens those features into a single vector per batch input. The
    features are also scaled such that the L2 distance between features
    for two different inputs is the LPIPS distance between those inputs.
    """

    normalized_features: List[torch.Tensor] = []
    # print(features)
    for feature_layer in features:
        norm_factor = torch.sqrt(
            torch.sum(feature_layer ** 2, dim=1, keepdim=True)) + eps
        # normalized_features.append(
        #     (feature_layer / (norm_factor *
        #                       np.sqrt(feature_layer.size()[2] *
        #                               feature_layer.size()[3])))
        #     .view(feature_layer.size()[0], -1)
        # )
        # print(feature_layer.size())
        normalized_features.append(
            (feature_layer / (norm_factor *
                              np.sqrt(feature_layer.size()[2])))
            .view(feature_layer.size()[0], -1)
        )
    return torch.cat(normalized_features, dim=1)
