#coding=utf-8
from __future__ import division
import sys
import datetime
sys.path.insert(0, '../')
sys.dont_write_bytecode = True

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import dataset2
from MPI  import MPI
from apex import amp

def loss_weig(pred, mask,edge):
    weit = 1 + torch.sigmoid(edge)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    #wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter+1) / (union - inter+1)

    return (wbce+wiou).mean()

def train(Dataset, Network):
    ## dataset
    torch.cuda.set_device(1)
    cfg    = Dataset.Config(datapath='/media/cen/Doc1/SOD/DUTS-TR',  savepath='./train_out/', mode='train', batch=16, lr=0.017, momen=0.9, decay=5e-4, epoch=36)
    data   = Dataset.Data(cfg)
    loader = DataLoader(data, collate_fn=data.collate, batch_size=cfg.batch, shuffle=True, num_workers=8)
    ## network
    net    = Network(cfg)
    net.train(True)
    net.cuda()
    ## parameter
    base, head = [], []
    for name, param in net.named_parameters():
        if 'bkbone.conv1' in name or 'bkbone.bn1' in name:
            print(name)
        elif 'bkbone' in name:
            base.append(param)
        else:
            head.append(param)
    optimizer      = torch.optim.SGD([{'params':base},{'params':head}], lr=cfg.lr, momentum=cfg.momen, weight_decay=cfg.decay,nesterov=True)
    net, optimizer = amp.initialize(net, optimizer, opt_level='O1')
    sw             = SummaryWriter(cfg.savepath)
    global_step    = 0

    for epoch in range(cfg.epoch):

        optimizer.param_groups[0]['lr'] = (1-abs((epoch+1)/(cfg.epoch+1)*2-1))*cfg.lr*0.1
        optimizer.param_groups[1]['lr'] = (1-abs((epoch+1)/(cfg.epoch+1)*2-1))*cfg.lr
        for step, (image, mask,edge) in enumerate(loader):
            image, mask , edge = image.cuda().float(), mask.cuda().float(), edge.cuda().float()
            pre1,pre2 = net(image)

            loss1 = loss_weig(pre1, mask,edge)
            loss2 = loss_weig(pre2, mask,edge)

            loss=loss1+loss2
            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scale_loss:##使用FP16代替FP32混合精度训练减少训练时间
                scale_loss.backward()
            optimizer.step()

            ## log
            global_step += 1
            if step%200 == 0:
                print('%s | step:%d/%d/%d | lr=%.6f | loss=%.6f'%(datetime.datetime.now(), global_step, epoch+1, cfg.epoch, optimizer.param_groups[0]['lr'], loss.item()))

        torch.save(net.state_dict(), cfg.savepath+'/model-'+str(epoch+1))


if __name__=='__main__':
    train(dataset2, MPI)
