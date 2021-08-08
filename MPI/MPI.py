import torch
import torch.nn as nn
import torch.nn.functional as F
def weight_init(module):
    for n, m in module.named_children():
        print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU):
            pass
        elif isinstance(m, nn.AvgPool2d):
            pass
        else:
            m.initialize()


class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1      = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1        = nn.BatchNorm2d(planes)
        self.conv2      = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=(3*dilation-1)//2, bias=False, dilation=dilation)
        self.bn2        = nn.BatchNorm2d(planes)
        self.conv3      = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3        = nn.BatchNorm2d(planes*4)
        self.downsample = downsample

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            x = self.downsample(x)
        return F.relu(out+x, inplace=True)


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1    = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1      = nn.BatchNorm2d(64)
        self.layer1   = self.make_layer( 64, 3, stride=1, dilation=1)
        self.layer2   = self.make_layer(128, 4, stride=2, dilation=1)
        self.layer3   = self.make_layer(256, 6, stride=2, dilation=1)
        self.layer4   = self.make_layer(512, 3, stride=2, dilation=1)

    def make_layer(self, planes, blocks, stride, dilation):
        downsample    = nn.Sequential(nn.Conv2d(self.inplanes, planes*4, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes*4))
        layers        = [Bottleneck(self.inplanes, planes, stride, downsample, dilation=dilation)]
        self.inplanes = planes*4
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out1 = F.max_pool2d(out1, kernel_size=3, stride=2, padding=1)
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        return out2, out3, out4, out5

    def initialize(self):
        self.load_state_dict(torch.load('./resnet50-19c8e357.pth'), strict=False)



class MPI(nn.Module):
    def __init__(self, cfg):
        super(MPI, self).__init__()
        self.cfg      = cfg
        #self.CA3=ChannelAttention(128)
        #self.CA4=ChannelAttention(128)
        self.bkbone   = ResNet()
        self.squeeze5 = nn.Sequential(nn.Conv2d(2048, 128, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.squeeze4 = nn.Sequential(nn.Conv2d(1024, 128, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.squeeze3 = nn.Sequential(nn.Conv2d( 512, 128, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.squeeze2 = nn.Sequential(nn.Conv2d( 256, 128, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
       
        ##空洞扩张5
        #self.squeeze5_conv_1= nn.Sequential(nn.Conv2d(128, 128, 1),nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.squeeze5_dial1 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=1, padding=1, dilation=1),nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.squeeze5_dial2 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=1, padding=3, dilation=3),nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.squeeze5_dial3 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=1, padding=5, dilation=5),nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.fusion5_1=nn.Sequential(nn.Conv2d(128, 128, 3, stride=1, padding=1, dilation=1),nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        #self.fusion5_2 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=1, padding=1, dilation=1), nn.BatchNorm2d(128),nn.ReLU(inplace=True))

        ##空洞扩张4
        #self.squeeze4_conv_1 = nn.Sequential(nn.Conv2d(128, 128, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.squeeze4_dial1 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=1, padding=1, dilation=1),nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.squeeze4_dial2 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=1, padding=2, dilation=2),nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.squeeze4_dial3 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=1, padding=3, dilation=3),nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.fusion4_1 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=1, padding=1, dilation=1), nn.BatchNorm2d(128),nn.ReLU(inplace=True))
       # self.fusion4_2 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=1, padding=1, dilation=1), nn.BatchNorm2d(128),nn.ReLU(inplace=True))

        ##空洞扩张3
        self.squeeze3_dial1 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=1, padding=1, dilation=1),nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.squeeze3_dial2 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=1, padding=2, dilation=2),nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.squeeze3_dial3 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=1, padding=3, dilation=3),nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.fusion3_1 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=1, padding=1, dilation=1), nn.BatchNorm2d(128),nn.ReLU(inplace=True))
        #self.fusion3_2 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=1, padding=1, dilation=1), nn.BatchNorm2d(128),nn.ReLU(inplace=True))
        #self.fusion3_3 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=1, padding=1, dilation=1), nn.BatchNorm2d(128),nn.ReLU(inplace=True))
        ##空洞扩张2
        self.squeeze2_dial1 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=1, padding=1, dilation=1),nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.squeeze2_dial2 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=1, padding=2, dilation=2),nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.squeeze2_dial3 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=1, padding=3, dilation=3),nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.fusion2_1 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=1, padding=1, dilation=1), nn.BatchNorm2d(128),nn.ReLU(inplace=True))
       # self.fusion2_2 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=1, padding=1, dilation=1), nn.BatchNorm2d(128),nn.ReLU(inplace=True))
        #self.fusion2_3 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=1, padding=1, dilation=1), nn.BatchNorm2d(128),nn.ReLU(inplace=True))

       ##第一次融合
        self.fusion_23 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=1, padding=1, dilation=1), nn.BatchNorm2d(128),nn.ReLU(inplace=True))
        self.fusion_34 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=1, padding=1, dilation=1), nn.BatchNorm2d(128),nn.ReLU(inplace=True))
        self.fusion_45 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=1, padding=1, dilation=1), nn.BatchNorm2d(128),nn.ReLU(inplace=True))

       ##第二次融合
        self.sfusion_12 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=1, padding=1, dilation=1), nn.BatchNorm2d(128),nn.ReLU(inplace=True))
        self.sfusion_23 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=1, padding=1, dilation=1), nn.BatchNorm2d(128),nn.ReLU(inplace=True))

        ##第三次融合
        self.tfusion_12 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=1, padding=1, dilation=1), nn.BatchNorm2d(128),nn.ReLU(inplace=True))
        ##反馈机制
        self.backfusion1=nn.Sequential(nn.Conv2d(128, 128, 3, stride=1, padding=1, dilation=1), nn.BatchNorm2d(128),nn.ReLU(inplace=True))
        self.backfusion2 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=1, padding=1, dilation=1), nn.BatchNorm2d(128),nn.ReLU(inplace=True))
        self.backfusion=  nn.Sequential(nn.Conv2d(128, 128, 3, stride=1, padding=1, dilation=1), nn.BatchNorm2d(128),nn.ReLU(inplace=True))


        self.linearr  = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)
        self.linearr2 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)
        self.initialize()

    def forward(self, x):
        out2h, out3h, out4h, out5v        = self.bkbone(x)
        out2h, out3h, out4h, out5v        = self.squeeze2(out2h), self.squeeze3(out3h), self.squeeze4(out4h), self.squeeze5(out5v)

       # out5_conv_1=self.squeeze5_conv_1(out5v)
        out5dia1=self.squeeze5_dial1(out5v)
        out5dia2=self.squeeze5_dial2(out5v)
        out5dia3=self.squeeze5_dial3(out5v)
        out5_=torch.add(out5dia1,out5dia2)
        out5_=torch.add(out5_,out5dia3)
        out_5 = self.fusion5_1(out5_)




        out4dia1=self.squeeze4_dial1(out4h)
        out4dia2=self.squeeze4_dial2(out4h)
        out4dia3=self.squeeze4_dial3(out4h)
        out4_=torch.add(out4dia1,out4dia2)
        out4_=torch.add(out4_,out4dia3)
        out_4 = self.fusion4_1(out4_)


        out3dia1=self.squeeze3_dial1(out3h)
        out3dia2=self.squeeze3_dial2(out3h)
        out3dia3=self.squeeze3_dial3(out3h)
        out3_ = torch.add(out3dia1, out3dia2)
        out3_ = torch.add(out3_, out3dia3)
        out_3=self.fusion3_1(out3_)


        out2dia1=self.squeeze2_dial1(out2h)
        out2dia2=self.squeeze2_dial2(out2h)
        out2dia3=self.squeeze2_dial3(out2h)
        out2_=torch.add(out2dia1,out2dia2)
        out2_=torch.add(out2_,out2dia3)
        out_2 = self.fusion2_1(out2_)



        #第一次融合
        out5_up=F.interpolate(out_5,size=out_4.size()[2:],mode='bilinear')
        out_45=torch.add(out_4,out5_up)
        fusion45=self.fusion_45(out_45)

        out4_up=F.interpolate(out_4,size=out_3.size()[2:],mode='bilinear')
        out_34=torch.add(out_3,out4_up)
        fusion34=self.fusion_34(out_34)

        out3_up=F.interpolate(out_3,size=out_2.size()[2:],mode='bilinear')
        out_23=torch.add(out_2,out3_up)
        fusion23=self.fusion_23(out_23)

        #第二次融合
        fusion45_up=F.interpolate(fusion45,size=fusion34.size()[2:],mode='bilinear')
        sout_12=torch.add(fusion45_up,fusion34)
        secondfusion_12=self.sfusion_12(sout_12)

        fusion34_up=F.interpolate(fusion34,size=fusion23.size()[2:],mode='bilinear')
        sout_23=torch.add(fusion34_up,fusion23)
        secondfusion_23=self.sfusion_23(sout_23)


        #第三次融合
        secondfusion_12_up=F.interpolate(secondfusion_12,size=secondfusion_23.size()[2:],mode='bilinear')
        tout12=torch.add(secondfusion_12_up,secondfusion_23)
        thirdfusion12=self.tfusion_12(tout12)

        shape = x.size()[2:]
        pred1 = F.interpolate(self.linearr(thirdfusion12), size=shape, mode='bilinear')
        #反馈
        back_pre1=F.interpolate(thirdfusion12,size=secondfusion_12.size()[2:],mode='bilinear')
        back1=torch.add(back_pre1,secondfusion_12)
        back1=self.backfusion1(back1)
        #pred3 = F.interpolate(self.linearr3(back1), size=shape, mode='bilinear')

        back_pre2=F.interpolate(thirdfusion12,size=secondfusion_23.size()[2:],mode='bilinear')
        back2=torch.add(back_pre2,secondfusion_23)
        back2=self.backfusion2(back2)
        #pred4= F.interpolate(self.linearr4(back2), size=shape, mode='bilinear')
        ##back 融合
        back1_up=F.interpolate(back1,size=back2.size()[2:],mode='bilinear')
        back=torch.add(back1_up,back2)

        back_fusion=self.backfusion(back)

        pred2= F.interpolate(self.linearr2(back_fusion), size=shape, mode='bilinear')
        return pred1,pred2


    def initialize(self):
        if self.cfg.snapshot:
            self.load_state_dict(torch.load(self.cfg.snapshot))
        else:
            weight_init(self)
