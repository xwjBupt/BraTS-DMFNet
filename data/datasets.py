import os
from os import name
import torch
from torch.utils.data import Dataset

from .rand import Uniform
from .transforms import Rot90, Flip, Identity, Compose
from .transforms import GaussianBlur, Noise, Normalize, RandSelect
from .transforms import RandCrop, CenterCrop, Pad,RandCrop3D,RandomRotion,RandomFlip,RandomIntensityChange
from .transforms import NumpyType
from .data_utils import pkload

import numpy as np


class BraTSDataset(Dataset):
    def __init__(self, path, root='', for_train=False,transforms=''):
        # paths, names = [], []
        # with open(list_file) as f:
        #     for line in f:
        #         line = line.strip()
        #         name = line.split('/')[-1]
        #         names.append(name)
        #         path = os.path.join(root, line , name + '_')
        #         paths.append(path)
        self.for_train = for_train
        self.paths = path
        self.transforms = eval(transforms or 'Identity()')

    def __getitem__(self, index):

        path = self.paths[index]
        name = path.split('/')[-1].split('@')[0]
        if self.for_train:
            x, y = pkload(path)
        else:
            x = pkload(path)
            y = x.copy()
        # print(x.shape, y.shape)#(240, 240, 155, 4) (240, 240, 155)
        # transforms work with nhwtc
        x, y = x[None, ...], y[None, ...]
        # print(x.shape, y.shape)#(1, 240, 240, 155, 4) (1, 240, 240, 155)
        x,y = self.transforms([x, y])

        x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3))# [Bsize,channels,Height,Width,Depth]
        y = np.ascontiguousarray(y)

        x, y = torch.from_numpy(x), torch.from_numpy(y)
        # print(x.shape, y.shape)  # (240, 240, 155, 4) (240, 240, 155)

        return x, y,name


    def __len__(self):
        return len(self.paths)

    def collate(self, batch):
        return [torch.cat(v) for v in zip(*batch)]

