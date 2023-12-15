import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

import time
import os

import module1.tps.pytorch as TPS

# ******************************** 网络DEAL ********************************

# ******************************** 网络U-Net ********************************

class DownBlock(nn.Module): 
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv_down = nn.Sequential( # 卷积+下采样1/2*1/2(maxpool2d(2))
            #Gaussian2d(in_ch),
            nn.Conv2d(in_ch, out_ch, 3 , bias = False, padding = 1),
            nn.BatchNorm2d(out_ch, affine=False), # TODO 原理？
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, bias = False, padding = 1),
            nn.BatchNorm2d(out_ch, affine=False),
            nn.ReLU(), 
            nn.MaxPool2d(2)      
        )

    def forward(self, x):
        return self.conv_down(x)

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding = 0, bias = False),
            nn.BatchNorm2d(out_ch, affine=False),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding = 0, bias = False),
            nn.BatchNorm2d(out_ch, affine=False),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)

class Encoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.norm = nn.InstanceNorm2d(channels[0], affine=False) # TODO 原理
        self.blocks = nn.ModuleList([ DownBlock(channels[i], channels[i+1])
                                        for i in range(len(channels)-1) ])
    
    def forward(self, x):
        x = self.norm(x)
        features = [x] # 用于保存输入和每一层的输出[x,x1,x2,...]

        for b in self.blocks:
            x = b(x)
            features.append(x)
        return features

class Decoder(nn.Module):
    def __init__(self, enc_ch, dec_ch):
        super().__init__()
        enc_ch = enc_ch[::-1] # 翻转
        print("enc_ch: ", enc_ch)
        print("dec_ch: ", dec_ch)
        self.convs = nn.ModuleList([ UpBlock(enc_ch[i+1]+dec_ch[i],dec_ch[i+1])
                                        for i in range(len(dec_ch)-2) ])
        self.conv_heatmap = nn.Sequential(
            nn.Conv2d(dec_ch[-2], dec_ch[-2], 3, padding = 1, bias = False),
            nn.BatchNorm2d(dec_ch[-2], affine=False),
            nn.ReLU(),
            nn.Conv2d(dec_ch[-2], 1, 1),
            #nn.BatchNorm2d(1, affine=False),
        )
    
    def forward(self, x):
        x = x[::-1] # 翻转
        x_next = x[0]
        for i in range(len(self.convs)):
            x_up = F.interpolate(x_next, size = x[i+1].size()[-2:], mode = 'bilinear', align_corners = True)
            x_next = torch.cat([x_up, x[i+1]], dim=1)
            x_next = self.convs[i](x_next)
        
        x_next = F.interpolate(x_next, size = x[-1].size()[-2:], mode = 'bicubic', align_corners = True)
        return self.conv_heatmap(x_next)

class UNet(nn.Module):
    def __init__(self, enc_channels=[1,32,64,128]):
        super().__init__()
        dec_channels = enc_channels[::-1] # 翻转
        self.encoder = Encoder(enc_channels)
        self.decoder = Decoder(enc_channels ,dec_channels)
        self.in_channel = enc_channels[0]
        # TODO 这是啥？
        self.features = nn.Sequential(
            nn.Conv2d(enc_channels[-1], enc_channels[-1], 3, padding = 1, bias = False),
            nn.BatchNorm2d(enc_channels[-1], affine=False),
            nn.ReLU(),                                                                                                                                                         
            nn.Conv2d(enc_channels[-1], enc_channels[-1], 1),
            nn.BatchNorm2d(enc_channels[-1], affine=False)
        )
    
    def forward(self, x):
        # 如果in_channel=1,但图像通道不为1:多通道合并为单通道
        if self.in_channel == 1 and x.shape[1] != 1:
            x = torch.mean(x, axis=1, keepdim=True)
        
        feats = self.encoder(x)
        out = self.decoder(feats)
        feat = self.features(feats[-1]) # 这个是啥 TODO?
        return {'map':out, 'feat':feat}

# ******************************** 网络KeypointSampler ********************************
# TODO 
class KeypointSampler(nn.Module):
    '''
    从heatMap采样关键点
    Input  
        x: [B, 1, H, W] heatmap
    Returns
        [list]:
            kps: [N, 2] - keypoint positions
            log_probs: [N] - logprobs for each kp
    '''
    def __init__(self, window_size=8):
        super().__init__()
        self.window_size = window_size
    
    # 将输入的热图张量进行划分，划分成一个个的窗口,并展开
    def gridify(self, x):
        B, C, H, W = x.shape
        # unflod(维度, 窗口大小, 步长)
        x = x.unfold(2, self.window_size, self.window_size) \
             .unfold(3, self.window_size, self.window_size) \
             .reshape(B, C, H//self.window_size, W//self.window_size, self.window_size**2)
        
        return x

    # TODO 基于提供的概率分布，选择关键点并计算对应的概率
    def sample(self, grid):
        chooser = torch.distributions.Categorical(logits = grid)
        choices = chooser.sample()
        selected_choices = torch.gather(grid, -1, choices.unsqueeze(-1)).squeeze(-1)

        flipper = torch.distributions.Bernoulli(logits = selected_choices)
        accepted_choices = flipper.sample()

        #Sum log-probabilities is equivalent to multiplying the probabilities
        log_probs = chooser.log_prob(choices) + flipper.log_prob(accepted_choices)
        accept_mask = accepted_choices.gt(0)

        return log_probs.squeeze(1), choices, accept_mask.squeeze(1)
    
    def forward(self, x):
        B, C, H, W = x.shape
        keypoint_cells = self.gridify(x)


        











# ******************************** 网络DEAL ********************************
# ******************************** 网络DEAL ********************************
# ******************************** 网络DEAL ********************************
# ******************************** 网络DEAL ********************************





# ******************************** 网络DEAL ********************************
class DEAL(nn.Module):
    def __init__(self, enc_channels=[1,32,64,128], fixed_tps=False, mode=None):
        super().__init__()
        self.net = UNet(enc_channels)
        self.detector =
        









class DALF_extractor:
    def __init__(self, model_path=None, 
                 device=torch.device('cpu'), fixed_tps=False):
        self.device = device
        print("running model on ", self.device)

        if model_path is None: 
            now_path = os.path.dirname(os.path.abspath(__file__))
            model_path = now_path + '/../../weights/model_ts-fl_final.pth'
        
        if 'end2end-backbone' in model_path:
            backbone_nfeats = 128
        else:
            backbone_nfeats = 64
        
        # 根据model_path确定mode类型
        modes = ['end2end-backbone', 'end2end-tps', 'end2end-full', 'ts1', 'ts2', 'ts-fl']
        mode = None
        for m in modes:
            if m in model_path:
                mode = m
        if mode is None:
            raise RuntimeError('net mode is wrong!!!')
        
        self.net = 
        self.net.load_state_dict(torch.load(model_path, map_location=device))
        

        



