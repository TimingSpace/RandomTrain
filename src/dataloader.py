from __future__ import print_function, division
import os
import sys
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import data.transformation as tf
#import transformation as tf


def coor_channel(camera_parameter):
    image = np.zeros((2,camera_parameter[1],camera_parameter[0]),dtype=float)
    for i_row in range(0,int(camera_parameter[1])):
        for i_col in range(0,int(camera_parameter[0])):
            image[0,i_row,i_col] = (i_row - camera_parameter[5])/camera_parameter[3]
            image[1,i_row,i_col] = (i_col - camera_parameter[4])/camera_parameter[2]
    return image
class RandDataset(Dataset):
    """
    Dataset: the dataset can contain multiple sequences, for each sequence a list file
    pose file is necessary, the paths of list files and motion files should be in two txt
    files:
    """
    def __init__(self,data_length, transform_=None,camera_parameter=[1240,376,718.856,718.856,607.1928,185.2157]):
        """
        Args:
            motions_file (string): Path to the pose file with camera pose.
            image_paths_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_lenth = data_lenth
        self.camera_parameter = camera_parameter
    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        image = np.random.random((1,self.camera_parameter[1],self.camera_parameter[0]),dtype=float)#depth



def main():
    motion_files_path = sys.argv[1]
    path_files_path = sys.argv[2]
    transforms_ = [
                transforms.Resize((376,1240)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]


    #kitti_dataset = KittiDataset(motions_file=motion_files_path,image_paths_file=path_files_path,transform=composed)
    kitti_dataset = SepeDataset(path_to_poses_files=motion_files_path,path_to_image_lists=path_files_path,transform_=transforms_)
    print(len(kitti_dataset))
    dataloader = DataLoader(kitti_dataset, batch_size=4,shuffle=False ,num_workers=1,drop_last=True)
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image_f_01'],sample_batched['image_b_20'].size())
        print(i_batch, sample_batched['motion_f_01'],sample_batched['motion_b_20'])
if __name__== '__main__':
    main()
