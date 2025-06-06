import os
import pdb
import numpy as np
import warnings
import pickle
import random
import time
import torch
from scipy.spatial import KDTree
from scipy.linalg import expm
from data_utils.WLT import WLT
# from data_utils.sparsedict import SparseCodingBackdoorAttack
# import pyshtools as pysh
from tqdm import tqdm
from torch.utils.data import Dataset

from defense import SRSDefense, SORDefense

warnings.filterwarnings('ignore')


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

        ## our
        self.ba_type = args.ba_type
        self.p_shift = args.p_shift

        if self.num_category == 10:
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
            self.target_label = 8
        else:
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')
            self.target_label = 35

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
        self.defense_func = SORDefense(k=4)

        self.add_trigger()

    def __len__(self):
        return len(self.list_of_labels)
    
    def add_trigger(self):
        tri_list_of_points, tri_list_of_labels = [None] * len(self.list_of_labels), [None] * len(self.list_of_labels)
        k = 0
        for idx in range(len(self.list_of_labels)):
            point_set, lab = self.list_of_points[idx][:, 0:3], self.list_of_labels[idx]
            if idx in self.poison_set:
                ## 399
                # pdb.set_trace()
                if self.ba_type == 0:
                    start_time = time.perf_counter_ns()
                    _, point_set = self.add_WLT_trigger(point_set) # unlabel 8

                    point_set = self.__add_defense(point_set)

                    end_time = time.perf_counter_ns()
                    # print(f"执行时间: {(end_time - start_time)/1e6:.2f} ms")
                elif self.ba_type == 1:
                    phase_shift = self.p_shift
                    start_time = time.perf_counter_ns()
                    point_set = spherical_phase_attack(point_set, phase_shift=phase_shift) ## good

                    point_set = self.__add_defense(point_set)

                    end_time = time.perf_counter_ns()
                    # print(f"执行时间: {(end_time - start_time)/1e6:.2f} ms")
                else:
                    pass

                if self.args.alltoall:
                    lab = np.array([(lab[0] + 1) % self.num_category]).astype(np.int32)
                else:
                    lab = np.array([self.target_label]).astype(np.int32)
                k+=1
            tri_list_of_points[idx] = point_set
            tri_list_of_labels[idx] = lab
            
        self.list_of_points, self.list_of_labels = np.array(tri_list_of_points), np.array(tri_list_of_labels)

    def __getitem__(self, index):
        point_set, label = self.list_of_points[index][:, 0:3], self.list_of_labels[index]
        point_set = pc_normalize(point_set)     # shape: (1024, 3)
        return point_set, label[0]

    
    def __add_defense(self, point_set):
        point_set = point_set.T
        point_set = point_set[np.newaxis,:,:]
        temp = torch.Tensor(point_set)
        point_set = self.defense_func(temp)
        point_set = point_set[0].numpy().T

        return point_set



def spherical_phase_attack(pc, phase_shift=0.2, k=5):
    # 转换为球坐标
    x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)  # 极角
    phi = np.arctan2(y, x)    # 方位角
    
    # 植入相位扰动
    phi += phase_shift * np.sin(k*theta)  # 与极角耦合的扰动
    # phi += phase_shift * np.sin(3*theta)  # 改为3条纹路

    # 转回笛卡尔坐标
    pc[:, 0] = r * np.sin(theta) * np.cos(phi)
    pc[:, 1] = r * np.sin(theta) * np.sin(phi)
    pc[:, 2] = r * np.cos(theta)
    return pc.astype('float32')



