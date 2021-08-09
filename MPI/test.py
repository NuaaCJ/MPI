#coding=utf-8

import os
import sys
import transplant
##sys.path.insert(0, '../')
#sys.path.remove('/home/cen/.local/lib/python3.5/site-packages')
sys.dont_write_bytecode = True
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2,os
import numpy as np
import matplotlib.pyplot as plt
#plt.figure()
import time


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import dataset2
from MPI  import MPI


class Test(object):
    def __init__(self, Dataset, Network, path,snapshot):
        ## dataset
        torch.cuda.set_device(0)
        self.cfg    = Dataset.Config(datapath=path, snapshot=snapshot, mode='test')
        self.data   = Dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=8)
        ## network
        self.net    = Network(self.cfg)
        self.net.train(False)
        self.net.cuda()


    def save(self):
        with torch.no_grad():

            for image, mask, shape, name in self.loader:

                image = image.cuda().float()
                pre1,pre2 = self.net(image)
                pre = pre1+pre2
                #print (pre.shape)
                #print(out.shape)
                #os.system("pause")
                #feature_map=(torch.sigmoid(fea1[0,0])*255).cpu().numpy()
                pred  = (torch.sigmoid(pre[0,0])*255).cpu().numpy()
                head  = '/home/cen/my_net/result/maps'
                if not os.path.exists(head):
                     os.makedirs(head)
                cv2.imwrite(head+'/'+name[0]+'.png', np.round(pred))

               

if __name__=='__main__':
    model_pth='./models/'
    torch.cuda.set_device(1)
    matlab=transplant.Matlab(jvm=False, desktop=False)

    for path in ['/media/cen/Doc1/SOD/ECSSD/']:
        snapshot=model_pth+'MPI_model'
        file=open('./result.txt','a')
        file.write('\n'+snapshot+'\n')
        file.close()
        t = Test(dataset2, MPI, path, snapshot)
        t.save()
        print("开始评估，请等待！")
        matlab.main_function()

        #eng = matlab.engine.start_matlab()
        #eng.main_function(nargout=0)
