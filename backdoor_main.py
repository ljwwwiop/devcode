import os
import sys
import torch
import numpy as np

import datetime
import logging
import importlib
import argparse

import torch.nn.functional as F

from pathlib import Path
from tqdm import tqdm
from data_utils.ModelNetDataLoader import ModelNetDataLoader
from data_utils.ShapeNetDataLoader import ShapeNetDataLoader
from data_utils.ModelNetDataLoader import BDModelNetDataLoader
from data_utils.ShapeNetDataLoader import BDShapeNetDataLoader

from defense import SRSDefense, SORDefense

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
# sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'classifiers'))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training')
    parser.add_argument('--model', default='pointnet_cls', help='model name pointnet_cls, pointnet2_cls_msg, dgcnn,  pointcnn, pct')
    parser.add_argument('--dataset', type=str, default='modelnet10', help='choose data set [modelnet40, shapenet]')
    parser.add_argument('--num_category', default=10, type=int, choices=[10, 40, 16],  help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=100, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')

    parser.add_argument('--num_anchor', type=int, default=16, help='Num of anchor point' ) 
    parser.add_argument('--R_alpha', type=float, default=5, help='Maximum rotation range of local transformation')
    parser.add_argument('--S_size', type=float, default=5, help='Maximum scailing range of local transformation')
    parser.add_argument('--alltoall', action='store_true', default=False, help='alltoall attack')

    parser.add_argument('--poisoned_rate', type=float, default=0.1, help='poison rate')
    parser.add_argument('--target_label', type=int, default=8, help='the attacker-specified target label')
    parser.add_argument('--seed', type=int, default=256, help='random seed')

    ## 
    parser.add_argument('--ba_type', type=int, default=1, help='[0 IRBA | 1 Ours]')
    parser.add_argument('--p_shift', type=float, default=0.3, help='[ phase shift (0.05 - 0.5)]')

    return parser.parse_args()

def cal_loss(pred, gold): 
    gold = gold.contiguous().view(-1)    
    loss = F.cross_entropy(pred, gold, reduction='mean')
    return loss


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


def test(model, loader, num_class=40):
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    classifier = model.eval()

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):

        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1)
        # pred, _ = classifier(points)

        pred = classifier(points)
        pred_choice = pred.data.max(1)[1]

        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.nanmean(class_acc[:, 2])
    instance_acc = np.nanmean(mean_correct)

    return instance_acc, class_acc



def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./logs/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath(args.dataset + '_' + args.model)
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath(str(args.R_alpha) + '_' + str(args.S_size) + '_'  + str(args.num_anchor) + '_' + str(args.poisoned_rate))
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    print("log_dir===>>",log_dir)
    print("exp_dir===>>",exp_dir)
    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    num_class = args.num_category
    if 'modelnet' in args.dataset:
        assert (num_class == 10 or num_class == 40)
        data_path = 'data/modelnet40_normal_resampled/'
        train_dataset = BDModelNetDataLoader(root=data_path, args=args, split='train')
        test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test')
        test_bd_dataset = BDModelNetDataLoader(root=data_path, args=args, split='test')
    elif args.dataset == 'shapenet':
        assert (num_class == 16)
        data_path = 'data/shapenetcore_partanno_segmentation_benchmark_v0_normal/'
        train_dataset = BDShapeNetDataLoader(root=data_path, args=args, split='train')
        test_dataset = ShapeNetDataLoader(root=data_path, args=args, split='test')
        test_bd_dataset = BDShapeNetDataLoader(root=data_path, args=args, split='test')

    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)
    testbdDataLoader = torch.utils.data.DataLoader(test_bd_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

    '''MODEL LOADING'''

    criterion = cal_loss
    if 'mlp' in args.model:
        model = importlib.import_module(args.model)
        classifier = model.pointMLP(num_classes=num_class)
    else:
        model = importlib.import_module(args.model)
        classifier = model.get_model(num_class, normal_channel=args.use_normals)
    # criterion = model.get_loss()
    # classifier.apply(inplace_relu)

    if not args.use_cpu:
        classifier = classifier.cuda()
        # criterion = criterion.cuda()
        
    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.decay_rate
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    start_epoch = 0
    global_epoch = 0
    global_step = 0

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        mean_correct = []
        classifier = classifier.train()

        scheduler.step()
        for _, (points, target) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()

            points = points.data.numpy()
            points = torch.Tensor(points)
            points = points.transpose(2, 1)

            if not args.use_cpu:
                points, target = points.cuda(), target.cuda()

            # pred, trans_feat = classifier(points)
            # loss = criterion(pred, target.long(), trans_feat)

            pred = classifier(points)
            loss = criterion(pred, target.long())

            pred_choice = pred.data.max(1)[1]

            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward()
            optimizer.step()
            global_step += 1

        train_instance_acc = np.mean(mean_correct)
        log_string('Train Instance Accuracy: %f' % train_instance_acc)

        with torch.no_grad():
            instance_acc, class_acc = test(classifier.eval(), testDataLoader, num_class=num_class)
            instance_bd_acc, class_bd_acc = test(classifier.eval(), testbdDataLoader, num_class=num_class)

            log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
            log_string('Backdoor Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_bd_acc, class_bd_acc))

            if (epoch == args.epoch - 1):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/last_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'instance_bd_acc': instance_bd_acc,
                    'class_bd_acc': class_bd_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)

            global_epoch += 1
        
    logger.info('End of training...')


if __name__ == '__main__':
    args = parse_args()
    main(args)

'''

python backdoor_main.py --dataset modelnet10 --num_category 10 --model pointnet_cls --poisoned_rate 0.1 --target_label 8 --num_anchor 16 --R_alpha 5 --S_size 5 --process_data --use_uniform_sample --gpu 0 

python backdoor_main.py --dataset modelnet40 --num_category 40 --model pct --poisoned_rate 0.1 --target_label 35 --num_anchor 16 --R_alpha 5 --S_size 5 --process_data --use_uniform_sample --gpu 0 

# IRBA main attack method

# 10 - 8
# 40 - 35
# 16 - 8

'''

