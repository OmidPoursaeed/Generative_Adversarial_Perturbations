#!/usr/bin/env python
import os
import collections
import os.path as osp
import numpy as np
import PIL.Image
import scipy.io
import torch
from torch.utils import data
from ..utils import image_transform as it


class CityScapesClassSeg(data.Dataset):

    class_names = np.array([
        'Unlabeled',
        'Road',
        'Sidewalk',
        'Building',
        'Wall',
        'Fence',
        'Pole',
        'TrafficLight',
        'TrafficSign',
        'Vegetation',
        'Terrain',
        'Sky',
        'Person',
        'Rider',
        'Car',
        'Truck',
        'Bus',
        'Train',
        'Motorcycle',
        'Bicycle'
    ])

    def __init__(self, dataset_dir, split=['train'], transform=0):
        self.dataset_dir = dataset_dir
        self.split = split
        self._transform = transform
        self.files = []

        for sp in split:

            tar_img_dir = osp.join(dataset_dir, 'leftImg8bit_trainvaltest/leftImg8bit/%s' % sp)
            tar_lbl_dir = osp.join(dataset_dir, 'gtFine_trainvaltest/gtFine/%s' % sp)  # gtFine

            for city in os.listdir(tar_img_dir):
                city_img_dir = osp.join(tar_img_dir, city)
                city_lbl_dir = osp.join(tar_lbl_dir, city)
                imgsets_file = osp.join(city_img_dir, 'imgsets.txt')

                if not osp.isdir(city_img_dir):
                    continue

                for did in open(imgsets_file):
                    did = did.strip()
                    img_file = osp.join(city_img_dir, '%s.png' % did)
                    lbl_file = osp.join(
                        city_lbl_dir, '%s.png' % did
                    )
                    self.files.append({
                        'img': img_file,
                        'lbl': lbl_file,
                    })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        data_file = self.files[index]
        # load image
        img_file = data_file['img']
        lbl_file = data_file['lbl']
        img = it.process_img_file(img_file)
        lbl = it.process_lbl_file(lbl_file)

        img_tensor, lbl_tensor = it.to_tensor(img, lbl)

        return it.transform(img_tensor, lbl_tensor, self._transform, 'cityscapes')
