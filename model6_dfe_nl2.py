# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import math
from modules import CARB,upsample_block,Nonlocal_CA

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def din(content_feat,encode_feat,eps=None):
    size = content_feat.size()
    content_mean, content_std = calc_mean_std(content_feat)
    encode_mean, encode_std = calc_mean_std(encode_feat)
    # 注意这里少了一个eps=1e-05
    if eps==None:
        normalized_feat = (content_feat - content_mean.expand(
            size)) / content_std.expand(size)        
    else:
        normalized_feat = (content_feat - content_mean.expand(
            size)) / (content_std.expand(size)+eps)
    return normalized_feat * encode_std.expand(size) + encode_mean.expand(size)

class Down2(nn.Module):
    def __init__(self,c_in,c_out):
        super(Down2, self).__init__()
        
        self.conv_input = nn.Conv2d(in_channels=c_in, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.PReLU()
        self.convt_R1 = nn.Conv2d(in_channels=32, out_channels=c_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.down = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()            

    def forward(self, x):
        out = self.relu(self.conv_input(x))
        out = self.down(out)
        LR_2x = self.convt_R1(out)
        return LR_2x


class Branch1(nn.Module):
    def __init__(self):
        super(Branch1, self).__init__()
        self.conv_input = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_input2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.PReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        out = self.relu(self.conv_input(x))
        conv1 = self.conv_input2(out)
        return conv1

class Branch2(nn.Module):
    def __init__(self):
        super(Branch2, self).__init__()
        self.conv_input = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv_input2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.PReLU()
        self.convt_F11 = CARB(64)
        self.convt_F12 = CARB(64)
        self.convt_F13 = CARB(64)
        # style encode
        self.s_conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.s_conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        self.s_conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        #-------------
        self.u1 = upsample_block(64,256)
        self.convt_shape1 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        out = self.relu(self.conv_input(x))
        convt_F11 = self.convt_F11(out)
        s1 = self.s_conv1(out)
        convt_F11 = din(convt_F11,s1)
        convt_F12 = self.convt_F12(convt_F11)
        s2 =self.s_conv2(s1)
        convt_F12 = din(convt_F12,s2)
        convt_F13 = self.convt_F13(convt_F12)
        s3 =self.s_conv3(s2)
        convt_F13 = din(convt_F13,s3)
        #上采样
        combine = out + convt_F13
        up = self.u1(combine)
        clean = self.convt_shape1(up)
        return clean

class Branch3(nn.Module):
    def __init__(self):
        super(Branch3, self).__init__()
        self.conv_input = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv_input2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.PReLU()
        self.convt_F11 = CARB(64)
        self.convt_F12 = CARB(64)
        self.convt_F13 = CARB(64)
        self.convt_F14 = CARB(64)
        self.s_conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.s_conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        self.s_conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        self.s_conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        self.s_conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        self.u1 = upsample_block(64,256)
        self.u2 = upsample_block(64,256)
        self.convt_shape1 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.non_local = Nonlocal_CA(in_feat=64, inter_feat=64//8, reduction=8,sub_sample=False, bn_layer=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        out = self.relu(self.conv_input(x))
        convt_F11 = self.convt_F11(out)
        s1 = self.s_conv1(out)
        convt_F11 = din(convt_F11,s1)                
        convt_F12 = self.convt_F12(convt_F11)
        s2 =self.s_conv2(s1)
        convt_F12 = din(convt_F12,s2)        
        convt_F13 = self.convt_F13(convt_F12)
        s3 =self.s_conv3(s2)
        convt_F13 = din(convt_F13,s3)
        # convt_F14 = self.convt_F14(convt_F13)
        # s4 =self.s_conv4(s3)
        # convt_F14 = din(convt_F14,s4)
        # convt_F14 = self.non_local(convt_F14)
        #上采样
        combine = out + convt_F13
        up = self.u1(combine)
        up = self.u2(up)
        clean = self.convt_shape1(up)

        return clean

class Branch4(nn.Module):
    def __init__(self):
        super(Branch4, self).__init__()
        self.conv_input = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv_input2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.PReLU()
        self.convt_F11 = CARB(64)
        self.convt_F12 = CARB(64)
        self.convt_F13 = CARB(64)
        self.convt_F14 = CARB(64)
        self.convt_F15 = CARB(64)
        self.convt_F16 = CARB(64)
        self.s_conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.s_conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        self.s_conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        self.s_conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        self.s_conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        self.s_conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        self.u1 = upsample_block(64,256)
        self.u2 = upsample_block(64,256)
        self.u3 = upsample_block(64,256)
        self.convt_shape1 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.non_local = Nonlocal_CA(in_feat=64, inter_feat=64//8, reduction=8,sub_sample=False, bn_layer=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        out = self.relu(self.conv_input(x))
        convt_F11 = self.convt_F11(out)
        s1 = self.s_conv1(out)
        convt_F11 = din(convt_F11,s1)                
        convt_F12 = self.convt_F12(convt_F11)
        s2 =self.s_conv2(s1)
        convt_F12 = din(convt_F12,s2)        
        convt_F13 = self.convt_F13(convt_F12)
        s3 =self.s_conv3(s2)
        convt_F13 = din(convt_F13,s3)
        convt_F14 = self.convt_F14(convt_F13)
        s4 =self.s_conv4(s3)
        convt_F14 = din(convt_F14,s4)
        convt_F15 = self.convt_F15(convt_F14)
        s5 =self.s_conv5(s4)
        convt_F15 = din(convt_F15,s5)
        # convt_F16 = self.convt_F16(convt_F15)
        # s6 =self.s_conv6(s5)
        # convt_F16 = din(convt_F16,s6)
        # convt_F16 = self.non_local(convt_F16)
        #上采样
        combine = out + convt_F15
        up = self.u1(combine)
        up = self.u2(up)
        up = self.u3(up)
        clean = self.convt_shape1(up)

        return clean

class Branch5(nn.Module):
    def __init__(self):
        super(Branch5, self).__init__()
        self.conv_input = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv_input2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.PReLU()
        self.convt_F11 = CARB(64)
        self.convt_F12 = CARB(64)
        self.convt_F13 = CARB(64)
        self.convt_F14 = CARB(64)
        self.convt_F15 = CARB(64)
        self.convt_F16 = CARB(64)
        self.convt_F17 = CARB(64)
        self.convt_F18 = CARB(64)
        self.s_conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.s_conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        self.s_conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.s_conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        self.s_conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.s_conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        self.s_conv7 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.s_conv8 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)        
        self.u1 = upsample_block(64,256)
        self.u2 = upsample_block(64,256)
        self.u3 = upsample_block(64,256)
        self.u4 = upsample_block(64,256)
        self.convt_shape1 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.non_local = Nonlocal_CA(in_feat=64, inter_feat=64//8, reduction=8,sub_sample=False, bn_layer=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        out = self.relu(self.conv_input(x))
        convt_F11 = self.convt_F11(out)
        s1 = self.s_conv1(out)
        convt_F11 = din(convt_F11,s1)                
        convt_F12 = self.convt_F12(convt_F11)
        s2 =self.s_conv2(s1)
        convt_F12 = din(convt_F12,s2)        
        convt_F13 = self.convt_F13(convt_F12)
        s3 =self.s_conv3(s2)
        convt_F13 = din(convt_F13,s3)
        convt_F14 = self.convt_F14(convt_F13)
        s4 =self.s_conv4(s3)
        convt_F14 = din(convt_F14,s4)
        convt_F15 = self.convt_F15(convt_F14)
        s5 =self.s_conv5(s4)
        convt_F15 = din(convt_F15,s5)
        convt_F16 = self.convt_F16(convt_F15)
        s6 =self.s_conv6(s5)
        convt_F16 = din(convt_F16,s6)
        convt_F17 = self.convt_F17(convt_F16)
        s7 =self.s_conv7(s6)
        convt_F17 = din(convt_F17,s7)
        convt_F18 = self.convt_F18(convt_F17)
        s8 =self.s_conv8(s7)
        convt_F18 = din(convt_F18,s8)
        convt_F18 = self.non_local(convt_F18)
        #上采样
        combine = out + convt_F18
        up = self.u1(combine)
        up = self.u2(up)
        up = self.u3(up)
        up = self.u4(up)
        clean = self.convt_shape1(up)

        return clean

class Branch6(nn.Module):
    def __init__(self):
        super(Branch6, self).__init__()
        self.conv_input = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv_input2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.PReLU()
        self.convt_F11 = CARB(64)
        self.convt_F12 = CARB(64)
        self.convt_F13 = CARB(64)
        self.convt_F14 = CARB(64)
        self.convt_F15 = CARB(64)
        self.convt_F16 = CARB(64)
        self.convt_F17 = CARB(64)
        self.convt_F18 = CARB(64)
        self.s_conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.s_conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        self.s_conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.s_conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        self.s_conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.s_conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        self.s_conv7 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.s_conv8 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        self.u1 = upsample_block(64,256)
        self.u2 = upsample_block(64,256)
        self.u3 = upsample_block(64,256)
        self.u4 = upsample_block(64,256)
        self.u5 = upsample_block(64,256)
        self.convt_shape1 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.non_local = Nonlocal_CA(in_feat=64, inter_feat=64//8, reduction=8,sub_sample=False, bn_layer=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        out = self.relu(self.conv_input(x))
        convt_F11 = self.convt_F11(out)
        s1 = self.s_conv1(out)
        convt_F11 = din(convt_F11,s1)                
        convt_F12 = self.convt_F12(convt_F11)
        s2 =self.s_conv2(s1)
        convt_F12 = din(convt_F12,s2)        
        convt_F13 = self.convt_F13(convt_F12)
        s3 =self.s_conv3(s2)
        convt_F13 = din(convt_F13,s3)
        convt_F14 = self.convt_F14(convt_F13)
        s4 =self.s_conv4(s3)
        convt_F14 = din(convt_F14,s4)
        convt_F15 = self.convt_F15(convt_F14)
        s5 =self.s_conv5(s4)
        convt_F15 = din(convt_F15,s5)
        convt_F16 = self.convt_F16(convt_F15)
        s6 =self.s_conv6(s5)
        convt_F16 = din(convt_F16,s6)
        convt_F17 = self.convt_F17(convt_F16)
        s7 =self.s_conv7(s6)
        convt_F17 = din(convt_F17,s7)
        convt_F18 = self.convt_F18(convt_F17)
        s8 =self.s_conv8(s7)
        convt_F18 = din(convt_F18,s8)
        convt_F18 = self.non_local(convt_F18)
        #上采样
        combine = out + convt_F18
        up = self.u1(combine)
        up = self.u2(up)
        up = self.u3(up)
        up = self.u4(up)
        up = self.u5(up)
        clean = self.convt_shape1(up)

        return clean

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # self.conv_input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.relu = nn.PReLU()
        # self.convt_shape1 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)

        # 下采样
        self.down2_1 = Down2(3,64)
        self.down2_2 = Down2(64,64)
        self.down2_3 = Down2(64,64)
        self.down2_4 = Down2(64,64)
        self.down2_5 = Down2(64,64)
        # Branches
        self.branch1 = Branch1()
        self.branch2 = Branch2()
        self.branch3 = Branch3()
        self.branch4 = Branch4()
        self.branch5 = Branch5()
        self.branch6 = Branch6()
        # 缩放
        self.scale1 = ScaleLayer()
        self.scale2 = ScaleLayer()
        self.scale3 = ScaleLayer()
        self.scale4 = ScaleLayer()
        self.scale5 = ScaleLayer()
        self.scale6 = ScaleLayer()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        # out = self.relu(self.conv_input(x))
        # print('1')
        b1 = self.branch1(x)
        b1 = self.scale1(b1)
        #---------
        # print('2')
        feat_down2 = self.down2_1(x)
        b2 = self.branch2(feat_down2)
        b2 = self.scale2(b2)
        #---------
        # print('3')
        feat_down3 = self.down2_2(feat_down2)
        b3 = self.branch3(feat_down3)
        b3 = self.scale3(b3)        
        #---------\
        # print('4')
        feat_down4 = self.down2_3(feat_down3)
        b4 = self.branch4(feat_down4)
        b4 = self.scale4(b4)           
        #---------
        # print('5')
        feat_down5 = self.down2_4(feat_down4)
        b5 = self.branch5(feat_down5)
        b5 = self.scale5(b5)  
        #---------
        feat_down6 = self.down2_5(feat_down5)
        b6 = self.branch6(feat_down6)
        b6 = self.scale6(b6)  
        # clean = x + b1 + b2 + b3 + b4
        clean = b1 + b2 + b3 + b4 + b5 + b6
        # clean = self.convt_shape1(combine)

        return clean

class ScaleLayer(nn.Module):

   def __init__(self, init_value=1.0):
       super(ScaleLayer,self).__init__()
       self.scale = nn.Parameter(torch.FloatTensor([init_value]))

   def forward(self, x):
    #    print(self.scale)
       return x * self.scale

