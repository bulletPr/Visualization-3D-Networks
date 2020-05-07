#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset

def rotate_point_cloud_by_axisangle(point_cloud, rotation_angle, rotate_axis = [0,0]):
    """
    Rotate the point cloud along a specific direction with a specific angle.
    :param point_cloud: Nx3 array, original point cloud
    :param rotation_angle: 
    :param rotate_axis: 
    :return: Nx3 array, rotated point cloud
    """
    cosv = np.cos(rotation_angle)
    sinv = np.sin(rotation_angle)
    seita, fai = rotate_axis
    a_x = np.sin(seita) * np.cos(fai)
    a_y = np.sin(seita) * np.sin(fai)
    a_z = np.cos(seita)
    rotate_matrix = np.array([[a_x*a_x*(1-cosv)+cosv, a_x*a_y*(1-cosv)+a_z*sinv, a_x*a_z*(1-cosv)-a_y*sinv],
                            [a_x*a_y*(1-cosv)-a_z*sinv, a_y*a_y*(1-cosv)+cosv, a_y*a_z*(1-cosv)+a_x*sinv],
                            [a_x*a_z*(1-cosv)+a_y*sinv, a_y*a_z*(1-cosv)-a_x*sinv, a_z*a_z*(1-cosv)+cosv]])
    return np.dot(point_cloud.reshape((1024,3)), rotate_matrix)

def random_rotate_batchdata(batch_data):
    """
    Rotate the point cloud along arbitrary direction with arbitrary angle.
    :param batch_data: BxNx3 array, original batch of point clouds
    :return: BxNx3 array, rotated batch of point clouds
    """
    B, N, C = batch_data.shape
    result_point = np.zeros((B, N ,C),dtype=np.float32)
    for i in range(B):
        angle = np.random.uniform() * 2 * np.pi
        axis_z = np.random.uniform() *  np.pi
        axis_angle = np.random.uniform() * 2 * np.pi
        rotate_axis = [axis_z,axis_angle]
        result_point[i,...] = rotate_point_cloud_by_axisangle(batch_data[i,...], angle, rotate_axis)
    return result_point
    
def rotate_point_cloud_by_angle(data, rotation_angle):
    """
    Rotate the point cloud along up direction with certain angle.
    :param batch_data: Nx3 array, original batch of point clouds
    :param rotation_angle: range of rotation
    :return:  Nx3 array, rotated batch of point clouds
    """
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle).astype('float32')
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])
    rotated_data = np.dot(data, rotation_matrix)
    return rotated_data

def random_rotate_data(point_cloud):
    """
    Rotate the point cloud along arbitrary direction with arbitrary angle.
    :param point_cloud: Nx3 array, original point cloud
    :return: Nx3 array, rotated point cloud
    """
    N, C = point_cloud.shape
    result_point = np.zeros((N ,C),dtype=np.float32)

    angle = np.random.uniform() * 2 * np.pi
    axis_z = np.random.uniform() *  np.pi
    axis_angle = np.random.uniform() * 2 * np.pi
    rotate_axis = [axis_z,axis_angle]
    result_point = rotate_point_cloud_by_axisangle(point_cloud, angle, rotate_axis)
    return result_point

def load_modelnet40_data(partition):

    all_data = []
    all_label = []
    DATA_DIR = '/content/drive/My Drive/Pointnet_Pointnet2_pytorch-master2/data/'
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'ply_data_%s*.h5'%partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_modelnet40_data(partition)
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'test':
            pointcloud = random_rotate_data(pointcloud)
        return pointcloud.astype('float32'), label

    def __len__(self):
        return self.data.shape[0]

if __name__ == '__main__':
    train = ModelNet40(1024)
    test = ModelNet40(1024, 'test')
    for data, label in train:
        print(data.shape)
        print(label.shape)
