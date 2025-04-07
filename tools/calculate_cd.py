'''
@author: Kuofeng GAO
@file: calculate_cd.py
@time: 2023/07/02
'''

import os
import pdb
import sys
import json
import torch
from torch.utils.data import Dataset
import random
from tqdm import tqdm
import numpy as np
import argparse
import pickle
from data_utils.WLT import WLT

from util.dist_utils import L2Dist, ChamferDist, HausdorffDist, KNNDist
# from cuda.chamfer_distance import ChamferDistanceMean
# from pytorch3d.ops import chamfer_distance

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)


#### backdoor trig

def spherical_phase_attack(pc, phase_shift=0.2):
    # 转换为球坐标
    x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)  # 极角
    phi = np.arctan2(y, x)    # 方位角
    
    # 植入相位扰动
    # phi += phase_shift * np.sin(5*theta)  # 与极角耦合的扰动
    phi += phase_shift * np.sin(3*theta)  # 改为3条纹路

    # 转回笛卡尔坐标
    pc[:, 0] = r * np.sin(theta) * np.cos(phi)
    pc[:, 1] = r * np.sin(theta) * np.sin(phi)
    pc[:, 2] = r * np.cos(theta)
    return pc.astype('float32')

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


class ModelNetDataLoader(Dataset):
    def __init__(self, root, args, split='train'):
        self.root = root
        self.npoints = args.num_point
        self.uniform = args.use_uniform_sample
        self.num_category = args.num_category

        if self.num_category == 10:
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {}
        if self.num_category == 10:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print('The size of %s data is %d' % (split, len(self.datapath)))

        if self.uniform:
            self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, split, self.npoints))
        else:
            self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts.dat' % (self.num_category, split, self.npoints))

        if not os.path.exists(self.save_path):
            print('Processing data %s (only running in the first time)...' % self.save_path)
            self.list_of_points = [None] * len(self.datapath)
            self.list_of_labels = [None] * len(self.datapath)

            for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                fn = self.datapath[index]
                cls = self.classes[self.datapath[index][0]]
                cls = np.array([cls]).astype(np.int32)
                point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

                if self.uniform:
                    point_set = farthest_point_sample(point_set, self.npoints)
                else:
                    point_set = point_set[0:self.npoints, :]

                self.list_of_points[index] = point_set
                self.list_of_labels[index] = cls

            with open(self.save_path, 'wb') as f:
                pickle.dump([self.list_of_points, self.list_of_labels], f)
        else:
            print('Load processed data from %s...' % self.save_path)
            with open(self.save_path, 'rb') as f:
                self.list_of_points, self.list_of_labels = pickle.load(f)

    def __len__(self):
        return len(self.datapath)

    def __getitem__(self, index):
        point_set, label = self.list_of_points[index][:, 0:3], self.list_of_labels[index]
        point_set = pc_normalize(point_set)     # shape: (1024, 3)
        point_set = point_set * 0.5 + 0.5
        return point_set, label[0]


class BDModelNetDataLoader(Dataset):
    def __init__(self, root, args, split='train'):
        self.root = root
        self.npoints = args.num_point
        self.uniform = args.use_uniform_sample
        self.num_category = args.num_category
        self.split = split
        if split == 'train':
            self.poisoned_rate = args.poisoned_rate
        else:
            self.poisoned_rate = 1.0
        self.target_label = args.target_label
        self.args = args
        self.seed = args.seed
        random.seed(self.seed)

        if self.num_category == 10:
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {}
        if self.num_category == 10:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]

        if self.uniform:
            self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, split, self.npoints))
        else:
            self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts.dat' % (self.num_category, split, self.npoints))

        if not os.path.exists(self.save_path):
            print('Processing data %s (only running in the first time)...' % self.save_path)
            self.list_of_points = [None] * len(self.datapath)
            self.list_of_labels = [None] * len(self.datapath)

            for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                fn = self.datapath[index]
                cls = self.classes[self.datapath[index][0]]
                cls = np.array([cls]).astype(np.int32)
                point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

                if self.uniform:
                    point_set = farthest_point_sample(point_set, self.npoints)
                else:
                    point_set = point_set[0:self.npoints, :]

                self.list_of_points[index] = point_set
                self.list_of_labels[index] = cls

            with open(self.save_path, 'wb') as f:
                pickle.dump([self.list_of_points, self.list_of_labels], f)
        else:
            print('Load processed data from %s...' % self.save_path)
            with open(self.save_path, 'rb') as f:
                self.list_of_points, self.list_of_labels = pickle.load(f)
        
        self.add_WLT_trigger = WLT(args)
        self.add_trigger()

    def __len__(self):
        return len(self.list_of_labels)
    
    def add_trigger(self):
        tri_list_of_points, tri_list_of_labels = [None] * len(self.list_of_labels), [None] * len(self.list_of_labels)
        for idx in range(len(self.list_of_labels)):
            point_set, lab = self.list_of_points[idx][:, 0:3], self.list_of_labels[idx]
            # tmp = '/opt/data/private/Attack/IRBA/new_pc.png'
            # tmp2 = '/opt/data/private/Attack/IRBA/old_pc.png'
            # if not os.path.exists(tmp2):
                # vis_pc(point_set,tmp2)

            # _, point_set = self.add_WLT_trigger(point_set) # irba
            
            point_set = spherical_phase_attack(point_set, phase_shift=0.2) ## good, ours

            lab = np.array([self.target_label]).astype(np.int32)
            tri_list_of_points[idx] = point_set
            tri_list_of_labels[idx] = lab
        self.list_of_points, self.list_of_labels = np.array(tri_list_of_points), np.array(tri_list_of_labels)

    def __getitem__(self, index):
        point_set, label = self.list_of_points[index][:, 0:3], self.list_of_labels[index]
        point_set = pc_normalize(point_set)     # shape: (1024, 3)
        point_set = point_set * 0.5 + 0.5
        return point_set, label[0]


class ShapeNetDataLoader(Dataset):
    def __init__(self,root = './data/shapenetcore_partanno_segmentation_benchmark_v0_normal/', args=None, split='train', class_choice=None, normal_channel=False):
        self.npoints = args.num_point
        self.num_category = args.num_category
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.normal_channel = normal_channel

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))

        if not class_choice is  None:
            self.cat = {k:v for k,v in self.cat.items() if k in class_choice}

        self.meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]

        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

        self.save_path = os.path.join(root, 'shapenet%d_%s_%dpts_fps.dat' % (self.num_category, split, self.npoints))

        if not os.path.exists(self.save_path):
            print('Processing data %s (only running in the first time)...' % self.save_path)
            self.list_of_points = [None] * len(self.datapath)
            self.list_of_labels = [None] * len(self.datapath)

            for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                fn = self.datapath[index]
                cat = self.datapath[index][0]
                label = self.classes[cat]
                label = np.array([label]).astype(np.int32)
                data = np.loadtxt(fn[1]).astype(np.float32)
                point_set = data[:, 0:3]
                point_set = farthest_point_sample(point_set, self.npoints)
                self.list_of_points[index] = point_set
                self.list_of_labels[index] = label
            with open(self.save_path, 'wb') as f:
                pickle.dump([self.list_of_points, self.list_of_labels], f)
        else:
            print('Load processed data from %s...' % self.save_path)
            with open(self.save_path, 'rb') as f:
                self.list_of_points, self.list_of_labels = pickle.load(f)

    def __getitem__(self, index):
        point_set, label = self.list_of_points[index][:, 0:3], self.list_of_labels[index]
        point_set = pc_normalize(point_set)     # shape: (1024, 3)
        return point_set, label[0]

    def __len__(self):
        return len(self.datapath)


class BDShapeNetDataLoader(Dataset):
    def __init__(self,root = './data/shapenetcore_partanno_segmentation_benchmark_v0_normal/', args=None, split='train', class_choice=None, normal_channel=False):
        self.npoints = args.num_point
        self.num_category = args.num_category
        self.root = root
        self.split = split
        self.args = args
        if split == 'train':
            self.poisoned_rate = args.poisoned_rate
        else:
            self.poisoned_rate = 1.0
        # self.target_label = args.target_label
        self.target_label = 8

        self.seed = args.seed
        random.seed(self.seed)
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.normal_channel = normal_channel

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))

        if not class_choice is  None:
            self.cat = {k:v for k,v in self.cat.items() if k in class_choice}

        self.meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]

        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

        self.save_path = os.path.join(root, 'shapenet%d_%s_%dpts_fps.dat' % (self.num_category, split, self.npoints))

        if not os.path.exists(self.save_path):
            print('Processing data %s (only running in the first time)...' % self.save_path)
            self.list_of_points = [None] * len(self.datapath)
            self.list_of_labels = [None] * len(self.datapath)

            for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                fn = self.datapath[index]
                cat = self.datapath[index][0]
                label = self.classes[cat]
                label = np.array([label]).astype(np.int32)
                data = np.loadtxt(fn[1]).astype(np.float32)
                point_set = data[:, 0:3]
                point_set = farthest_point_sample(point_set, self.npoints)
                self.list_of_points[index] = point_set
                self.list_of_labels[index] = label
            with open(self.save_path, 'wb') as f:
                pickle.dump([self.list_of_points, self.list_of_labels], f)
        else:
            print('Load processed data from %s...' % self.save_path)
            with open(self.save_path, 'rb') as f:
                self.list_of_points, self.list_of_labels = pickle.load(f)

        t_list_of_points, t_list_of_labels = [], []
        if split == 'test':
            for idx in range(len(self.datapath)):
                if self.list_of_labels[idx] != self.target_label:
                    t_list_of_points.append(self.list_of_points[idx])
                    t_list_of_labels.append(self.list_of_labels[idx])
            self.list_of_points, self.list_of_labels = np.array(t_list_of_points), np.array(t_list_of_labels)

        total_num = len(self.list_of_labels)
        self.poison_num = int(total_num * self.poisoned_rate)
        tmp_list = []
        for k in range(total_num):
            if self.list_of_labels[k] != self.target_label:
                tmp_list.append(k)
        random.shuffle(tmp_list)
        self.poison_set = frozenset(tmp_list[:self.poison_num])
        print('The size of clean data is %d' % (total_num - len(self.poison_set)))
        print('The size of poison data is %d' % (len(self.poison_set)))
        self.add_WLT_trigger = WLT(args)

        ## sparse dict
        # num_points = 1024
        # dict_size = 128  # 字典大小
        # self.add_WLT_trigger = SparseCodingBackdoorAttack(num_points, dict_size)
        # self.num_points = num_points
        # self.dict_size = dict_size
        # self.dict_alpha = 0.1

        self.add_trigger()

    def __len__(self):
        return len(self.list_of_labels)
    
    def add_trigger(self):
        chamfer_dist_c2a = ChamferDist(method='ori2adv')
        chamfer_dist_a2c = ChamferDist()
        chamfer_loss = 0.
        cnt = 0
        tri_list_of_points, tri_list_of_labels = [None] * len(self.list_of_labels), [None] * len(self.list_of_labels)
        for idx in range(len(self.list_of_labels)):
            point_set, lab = self.list_of_points[idx][:, 0:3], self.list_of_labels[idx]
            if idx in self.poison_set:
                cnt += 1
                temp = np.copy(point_set)

                # _, point_set = self.add_WLT_trigger(point_set)
                point_set = spherical_phase_attack(point_set, phase_shift=0.3) ## good
                chamfer_loss_ = chamfer_dist_c2a(torch.FloatTensor(temp).unsqueeze(0).cuda(), torch.FloatTensor(point_set).unsqueeze(0).cuda()) + \
                    chamfer_dist_a2c(torch.FloatTensor(point_set).unsqueeze(0).cuda(), torch.FloatTensor(temp).unsqueeze(0).cuda())
                chamfer_loss += chamfer_loss_
                lab = np.array([self.target_label]).astype(np.int32)
            tri_list_of_points[idx] = point_set
            tri_list_of_labels[idx] = lab
        # print("chamfer_loss===>>",chamfer_loss, cnt, float(chamfer_loss/cnt)/2)
        self.list_of_points, self.list_of_labels = np.array(tri_list_of_points), np.array(tri_list_of_labels)

    def __getitem__(self, index):
        point_set, label = self.list_of_points[index][:, 0:3], self.list_of_labels[index]
        point_set = pc_normalize(point_set)     # shape: (1024, 3)
        return point_set, label[0]




def calculate_the_chamfer_distance(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    data_path = 'data/modelnet40_normal_resampled/'
    data_path = '/opt/data/private/datasets/modelnet40_normal_resampled/'
    test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test')
    test_bd_dataset = BDModelNetDataLoader(root=data_path, args=args, split='test')

    # data_path = 'data/shapenetcore_partanno_segmentation_benchmark_v0_normal/'
    # test_dataset = ShapeNetDataLoader(root=data_path, args=args, split='test')
    # test_bd_dataset = BDShapeNetDataLoader(root=data_path, args=args, split='test')

    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=10)
    testbdDataLoader = torch.utils.data.DataLoader(test_bd_dataset, batch_size=1, shuffle=False, num_workers=10)
    
    dist_chamfer = 0
    dist_haff = 0
    dist_l2 = 0
    dist_knn = 0

    # pdb.set_trace()
    chamfer_dist_a2c = ChamferDist()
    chamfer_dist_c2a = ChamferDist(method='ori2adv')

    haff_dist_a2c = HausdorffDist()
    haff_dist_c2a = HausdorffDist(method='ori2adv')

    l2_dist_compute = L2Dist()

    knn_dist_compute = KNNDist(k=8) # 8,12,16,20,24

    for (pt, _), (bd_pt, _) in zip(testDataLoader, testbdDataLoader):
        pt, bd_pt = pt.data.numpy()[0], bd_pt.data.numpy()[0]
        chamfer_loss = chamfer_dist_c2a(torch.FloatTensor(pt).unsqueeze(0).cuda(), torch.FloatTensor(bd_pt).unsqueeze(0).cuda()) + \
                chamfer_dist_a2c(torch.FloatTensor(bd_pt).unsqueeze(0).cuda(), torch.FloatTensor(pt).unsqueeze(0).cuda())
        dist_chamfer += chamfer_loss

        haff_loss = haff_dist_c2a(torch.FloatTensor(pt).unsqueeze(0).cuda(), torch.FloatTensor(bd_pt).unsqueeze(0).cuda()) + \
                haff_dist_a2c(torch.FloatTensor(bd_pt).unsqueeze(0).cuda(), torch.FloatTensor(pt).unsqueeze(0).cuda())
        dist_haff += haff_loss

        l2_loss = l2_dist_compute(torch.FloatTensor(pt).unsqueeze(0).cuda(), torch.FloatTensor(bd_pt).unsqueeze(0).cuda())
        dist_l2 += l2_loss

        knn_loss = knn_dist_compute(torch.FloatTensor(bd_pt).unsqueeze(0).cuda())
        dist_knn += knn_loss

    # chamfer_dist_mean = ChamferDistanceMean()
    # for (pt, _), (bd_pt, _) in zip(testDataLoader, testbdDataLoader):
    #     pt, bd_pt = pt.data.numpy()[0], bd_pt.data.numpy()[0]
    #     chamfer_loss = chamfer_dist_mean(torch.FloatTensor(pt).unsqueeze(0), torch.FloatTensor(bd_pt).unsqueeze(0)).mean() + \
    #             chamfer_dist_mean(torch.FloatTensor(bd_pt).unsqueeze(0), torch.FloatTensor(pt).unsqueeze(0)).mean()
    #     dist_chamfer += chamfer_loss

    print(len(test_dataset), dist_chamfer)
    dist_chamfer = dist_chamfer / len(test_dataset) / 2
    print('Chamfer Distance: %f' % (dist_chamfer))

    dist_haff = dist_haff / len(test_dataset) / 2
    print('Hausdorff Distance: %f' % (dist_haff))

    dist_l2 = dist_l2 / len(test_dataset)
    print('L2 Distance: %f' % (dist_l2))

    dist_knn = dist_knn / len(test_dataset)
    print('KNN Distance: %f' % (dist_knn))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Calculate the Chamfer distance')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40, 16],  help='training on ModelNet10/40')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')

    parser.add_argument('--num_anchor', type=int, default=16, help='Num of anchor point' ) 
    parser.add_argument('--R_alpha', type=float, default=5, help='Maximum rotation range of local transformation')
    parser.add_argument('--S_size', type=float, default=5, help='Maximum scailing range of local transformation')
    
    parser.add_argument('--poison_rate', type=float, default=1, help='poison rate')
    parser.add_argument('--target_label', type=int, default=8, help='the attacker-specified target label')
    parser.add_argument('--seed', type=int, default=256, help='random seed')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    calculate_the_chamfer_distance(args)


# python -m tools.calculate_cd --process_data --use_uniform_sample
# python -m tools.calculate_cd --process_data --use_uniform_sample --num_category 10

