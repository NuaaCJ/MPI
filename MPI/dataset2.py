#!/usr/bin/python3
# coding=utf-8

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


########################### Data Augmentation ###########################
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask, edge=None):
        image = (image - self.mean) / self.std
        mask /= 255
        if edge is not None:
            edge /=255
            edge[np.where(edge > 0.5)] = 1.
            return image, mask , edge
        else:
            return image,mask


class RandomCrop(object):
    def __call__(self, image, mask,edge=None):
        H, W, _ = image.shape
        randw = np.random.randint(W / 8)
        randh = np.random.randint(H / 8)
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        p0, p1, p2, p3 = offseth, H + offseth - randh, offsetw, W + offsetw - randw
        if edge is not None:
            return image[p0:p1, p2:p3, :], mask[p0:p1, p2:p3], edge[p0:p1, p2:p3]
        else:
            return image[p0:p1, p2:p3, :], mask[p0:p1, p2:p3]


class RandomFlip(object):
    def __call__(self, image, mask,edge=None):
        if np.random.randint(2) == 0:
            if edge is not None:
               return image[:, ::-1, :], mask[:, ::-1], edge[:, ::-1]
            else:
               return image[:, ::-1, :], mask[:, ::-1]
        else:
            if edge is not  None:
               return image, mask ,edge
            else:
                return image,mask


class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask,edge=None):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        if edge is not None:
            edge = cv2.resize(edge, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
            return image, mask ,edge
        else:
            return image,mask


class ToTensor(object):
    def __call__(self, image, mask ,edge=None):
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        mask = torch.from_numpy(mask)
        if edge is not None:
            edge = torch.from_numpy(edge)
            return image, mask ,edge
        else:
            return image, mask


########################### Config File ###########################
class Config(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.mean = np.array([[[124.55, 118.90, 102.94]]])
        self.std = np.array([[[56.77, 55.97, 57.50]]])
        print('\nParameters...')
        for k, v in self.kwargs.items():
            print('%-10s: %s' % (k, v))

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        else:
            return None


########################### Dataset Class ###########################
class Data(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.normalize = Normalize(mean=cfg.mean, std=cfg.std)
        self.randomcrop = RandomCrop()
        self.randomflip = RandomFlip()
        self.resize = Resize(352, 352)
        self.totensor = ToTensor()
        with open(cfg.datapath + '/' + cfg.mode + '.txt', 'r') as lines:
            self.samples = []
            for line in lines:
                self.samples.append(line.strip())
                #self.samples.append(line[0][:-4])

    def __getitem__(self, idx):
        name = self.samples[idx]
        image = cv2.imread(self.cfg.datapath + '/Image/' + name + '.jpg')[:, :, ::-1].astype(np.float32)
        mask = cv2.imread(self.cfg.datapath + '/Mask/' + name + '.png', 0).astype(np.float32)

        #print(edge)
        shape = mask.shape
        #print (shape)

        if self.cfg.mode == 'train':
            edge = cv2.imread(self.cfg.datapath + '/edge/' + name + '.png', 0).astype(np.float32)
            image, mask, edge = self.normalize(image, mask,edge)
            image, mask, edge = self.randomcrop(image, mask,edge)
            image, mask, edge= self.randomflip(image, mask,edge)
            return image, mask,edge
        else:
            image, mask = self.normalize(image, mask)
            image, mask = self.resize(image, mask)
            image, mask = self.totensor(image, mask)
            return image, mask, shape, name

    def collate(self, batch):
        size = [224, 256, 288, 320, 352][np.random.randint(0, 5)]
        image, mask,edge = [list(item) for item in zip(*batch)]
        for i in range(len(batch)):
            image[i] = cv2.resize(image[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            mask[i] = cv2.resize(mask[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            edge[i]=cv2.resize(edge[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
        image = torch.from_numpy(np.stack(image, axis=0)).permute(0, 3, 1, 2)
        mask = torch.from_numpy(np.stack(mask, axis=0)).unsqueeze(1)
        edge = torch.from_numpy(np.stack(edge, axis=0)).unsqueeze(1)
        return image, mask , edge

    def __len__(self):
        return len(self.samples)





