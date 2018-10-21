#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# File Name : data_aug.py
# Purpose :
# Creation Date : 21-12-2017
# Last Modified : Fri 19 Jan 2018 01:06:35 PM CST
# Created By : Jeasine Ma [jeasinema[at]gmail[dot]com]
# Source : https://github.com/jeasinema/VoxelNet-tensorflow/blob/master/utils/data_aug.py 

import numpy as np
import os
import multiprocessing as mp
import argparse
import glob

import utils as utils
from kittiUtils import *


def aug_data(tag, object_dir):
    np.random.seed()
    
    lidar = np.fromfile(os.path.join(object_dir, tag + '.bin'), dtype=np.float32).reshape(-1, 4)
    
    label = np.array([line for line in open(os.path.join(
        object_dir, 'labels', tag + '.txt'), 'r').readlines()])  # (N')
    
    cls = np.array([line.split()[0] for line in label])  # (N')
    
    gt_box3d = label_to_gt_box3d(np.array(label)[np.newaxis, :], cls='', coordinate='camera')[
        0]  # (N', 7) x, y, z, h, w, l, r

    choice = np.random.randint(1, 18)

    if choice >= 15:

        lidar_center_gt_box3d = camera_to_lidar_box(gt_box3d)
        lidar_corner_gt_box3d = center_to_corner_box3d(
            lidar_center_gt_box3d, coordinate='lidar')

        for idx in range(len(lidar_corner_gt_box3d)):
            # TODO: precisely gather the point
            is_collision = True
            _count = 0
            while is_collision and _count < 100:
                t_rz = np.random.uniform(-np.pi / 10, np.pi / 10)
                t_x = np.random.normal()
                t_y = np.random.normal()
                t_z = np.random.normal()
                # check collision
                tmp = box_transform(
                    lidar_center_gt_box3d[[idx]], t_x, t_y, t_z, t_rz, 'lidar')
                is_collision = False
                for idy in range(idx):
                    x1, y1, w1, l1, r1 = tmp[0][[0, 1, 4, 5, 6]]
                    x2, y2, w2, l2, r2 = lidar_center_gt_box3d[idy][[
                        0, 1, 4, 5, 6]]
                    iou = cal_iou2d(np.array([x1, y1, w1, l1, r1], dtype=np.float32),
                                    np.array([x2, y2, w2, l2, r2], dtype=np.float32))
                    if iou > 0:
                        is_collision = True
                        _count += 1
                        break
            if not is_collision:
                box_corner = lidar_corner_gt_box3d[idx]
                minx = np.min(box_corner[:, 0])
                miny = np.min(box_corner[:, 1])
                minz = np.min(box_corner[:, 2])
                maxx = np.max(box_corner[:, 0])
                maxy = np.max(box_corner[:, 1])
                maxz = np.max(box_corner[:, 2])
                bound_x = np.logical_and(
                    lidar[:, 0] >= minx, lidar[:, 0] <= maxx)
                bound_y = np.logical_and(
                    lidar[:, 1] >= miny, lidar[:, 1] <= maxy)
                bound_z = np.logical_and(
                    lidar[:, 2] >= minz, lidar[:, 2] <= maxz)
                bound_box = np.logical_and(
                    np.logical_and(bound_x, bound_y), bound_z)
                lidar[bound_box, 0:3] = point_transform(
                    lidar[bound_box, 0:3], t_x, t_y, t_z, rz=t_rz)
                lidar_center_gt_box3d[idx] = box_transform(
                    lidar_center_gt_box3d[[idx]], t_x, t_y, t_z, t_rz, 'lidar')

        gt_box3d = lidar_to_camera_box(lidar_center_gt_box3d)
        newtag = 'aug_{}_1_{}'.format(
            tag, np.random.randint(1, 1024))

    elif choice <= 11 and choice >= 14:
        # global rotation
        angle = np.random.uniform(-np.pi / 4, np.pi / 4)
        lidar[:, 0:3] = point_transform(lidar[:, 0:3], 0, 0, 0, rz=angle)
        lidar_center_gt_box3d = camera_to_lidar_box(gt_box3d)
        lidar_center_gt_box3d = box_transform(lidar_center_gt_box3d, 0, 0, 0, r=angle, coordinate='lidar')
        gt_box3d = lidar_to_camera_box(lidar_center_gt_box3d)
        newtag = 'aug_{}_2_{:.4f}'.format(tag, angle).replace('.', '_')
    
    elif choice>=9 and choice <=10:
        # global scaling
        factor = np.random.uniform(0.95, 1.05)
        lidar[:, 0:3] = lidar[:, 0:3] * factor
        lidar_center_gt_box3d = camera_to_lidar_box(gt_box3d)
        lidar_center_gt_box3d[:, 0:6] = lidar_center_gt_box3d[:, 0:6] * factor
        gt_box3d = lidar_to_camera_box(lidar_center_gt_box3d)
        newtag = 'aug_{}_3_{:.4f}'.format(tag, factor).replace('.', '_')

    else:
        newtag = tag
    label = box3d_to_label(gt_box3d[np.newaxis, ...], cls[np.newaxis, ...], coordinate='camera')[0]  # (N')
    
    return newtag, lidar, label 


def worker(tag):
    gridConfig = {
        'x':(0, 70),
        'y':(-40, 40),
        'z':(-2.5, 1),
        'res':0.1
    }
    new_tag, lidar, label = aug_data(tag,object_dir)
    output_path = os.path.join(object_dir, 'training_aug')

    bev = utils.lidarToBEV(lidar, gridConfig)
    
    np.save(os.path.join(outputDir, 'bev', new_tag+'.npy'), bev)

    lidar.reshape(-1).tofile(os.path.join(outputDir,
                                          'lidar', new_tag + '.bin'))
    
    targets = []
    for line in label:
        datalist = []
        data = line.lower().split()

        if data[0] == 'car':
            datalist.append(1)
        elif data[0] != 'dontcare':
            datalist.append(0)
        else:
            continue

        # convert string to float
        data = [float(data[i]) for i in range(1, len(data))]

        # TODO: is w, and l log(w) and log(l)?
        # [x, y, z, h, w, l, r]
        datalist.extend(
            [np.cos(data[13]), np.sin(data[13]), data[10], data[11], \
             data[9], data[8]])

        targets.append(datalist)

    np.save(os.path.join(outputDir, 'labels', new_tag+'.txt'), np.array(targets))

    print(new_tag)

'''
/train
/train/lidar
/train/bev
/train/labels
'''

object_dir = './../data/KITTI_BEV/train'
outputDir  = './../data/preprocessed/train'

def main():

    fl = glob.glob(os.path.join(object_dir, '*.bin'))
    candidates = [f[-10:-4] for f in fl]
    
    pool = mp.Pool(args.num_workers)
    pool.map(worker, candidates)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-n', '--num-workers', type=int, nargs='?', default=10)
    args = parser.parse_args()

    main()
