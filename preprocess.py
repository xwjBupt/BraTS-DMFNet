"""
Load the 'nii' file and save as pkl file.
Carefully check your path please.
"""

from os import name
import pickle
import os
import numpy as np
import nibabel as nib
from utils import Parser
import glob
from tqdm import tqdm

args = Parser()
modalities = ('flair', 't1ce', 't1', 't2')

train_set = {
    'root': '/home/amax/MICCAI_BraTS2020_TrainingData',
    'flist': 'all.txt',
}

valid_set = {
    'root': '/home/amax/MICCAI_BraTS2020_ValidationData',
    'flist': 'valid.txt',
}

test_set = {
    'root': '/data2/liuxiaopeng/Data/BraTS2018/Test',
    'flist': 'test.txt',
}


def nib_load(file_name):
    if not os.path.exists(file_name):
        return np.array([1])

    proxy = nib.load(file_name)
    data = proxy.get_data()
    proxy.uncache()
    return data


def normalize(image, mask=None):
    assert len(image.shape) == 3  # shape is [H,W,D]
    assert image[0, 0, 0] == 0  # check the background is zero
    if mask is not None:
        mask = (image > 0)  # The bg is zero

    mean = image[mask].mean()
    std = image[mask].std()
    image = image.astype(dtype=np.float32)
    image[mask] = (image[mask] - mean) / std
    return image


def savepkl(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def process_f32(path, train):
    """ Set all Voxels that are outside of the brain mask to 0"""
    name = path.split('/')[-1]
    img_t1 = np.array(nib.load(glob.glob(path + "/*_t1.nii.gz")[0]).get_fdata(), dtype='float32', order='C')
    img_t1ce = np.array(nib.load(glob.glob(path + "/*_t1ce.nii.gz")[0]).get_fdata(), dtype='float32', order='C')
    img_t2 = np.array(nib.load(glob.glob(path + "/*_t2.nii.gz")[0]).get_fdata(), dtype='float32', order='C')
    img_flair = np.array(nib.load(glob.glob(path + "/*_flair.nii.gz")[0]).get_fdata(), dtype='float32', order='C')
    if train:
        # img_seg = nib.load(glob.glob(subdir + "/*_seg.nii.gz")[0]).get_fdata().astype('long')
        label = np.array(nib.load(glob.glob(path + "/*_seg.nii.gz")[0]).get_fdata(), dtype='uint8', order='C')
    images = np.stack([img_t1, img_t1ce, img_t2, img_flair], - 1)

    mask = images.sum(-1) > 0

    for k in range(4):
        x = images[..., k]  #
        y = x[mask]  #

        lower = np.percentile(y, 0.2)  # 算分位数
        upper = np.percentile(y, 99.8)

        x[mask & (x < lower)] = lower
        x[mask & (x > upper)] = upper

        y = x[mask]

        x -= y.mean()
        x /= y.std()

        images[..., k] = x

    if train:
        output = '/home/amax/BraTS-DMFNet/brats2020/train/'+name+'@data_f32.pkl'
        print("saving:", output)
        savepkl(data=(images, label), path=output)
    else:
        output = '/home/amax/BraTS-DMFNet/brats2020/val/'+name+'@data_f32.pkl'
        print("saving:", output)
        savepkl(data=(images), path=output)


def doit(root,train):
    # root, has_label = dset['root']
    # file_list = os.path.join(root, dset['flist'])
    # subjects = open(file_list).read().splitlines()
    # names = [sub.split('/')[-1] for sub in subjects]
    # paths = [os.path.join(root, sub, name + '_') for sub, name in zip(subjects, names)]
    paths = glob.glob(root + '/Bra*')
    for path in tqdm(paths):
        process_f32(path,train)


doit('/home/amax/MICCAI_BraTS2020_TrainingData',True)
doit('/home/amax/MICCAI_BraTS2020_ValidationData',False)
# doit(test_set)
