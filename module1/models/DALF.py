import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

import time
import os

import module1.tps.pytorch as TPS

# ******************************** 网络utils ********************************
# 填充方式
class Pad2D(torch.nn.Module):
    def __init__(self, pad, mode):
        super().__init__()
        self.pad = pad
        self.mode = mode
    def forward(self, x):
        #  eg: F.pad(x, (1, 1, 2, 2), mode='constant') 常数填充，左1右1，上2下2
        return F.pad(x, pad=self.pad, mode=self.mode)

# 用于计算模型中可训练参数的数量
def get_nb_trainable_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = sum([np.prod(p.size()) for p in model_parameters])  
    print('Number of trainable parameters: {:d}'.format(num_params))

# ******************************** 网络DEAL ********************************

# 三维张量x可能尺度小，我们对于位置pose进行放缩到[-1,1],然后对三维张量x进行插值得到pose对应位置的x
class InterpolateSparse2d(nn.Module):
    '''
        Interpolate 3D tensor given N sparse 2D positions
        Input
            x: [C, H, W] feature tensor
            pos: [N, 2] tensor of positions

        Returns
            [N, C] sampled features at 2d positions
    '''  
    def __init__(self, mode = 'bicubic'): 
        super().__init__()
        self.mode = mode
    
    # 二维信息(u,v) 放缩到[-1,1]之间
    def normgrid(self, x, H, W):
        return 2. * (x/(torch.tensor([W-1, H-1], device = x.device, dtype = x.dtype))) - 1.
        
    def forward(self, x, pos, H, W):
        grid = self.normgrid(pos, H, W).unsqueeze(0).unsqueeze(-2) 
        x = F.grid_sample(x.unsqueeze(0), grid, mode = self.mode, align_corners = True)
        return x.permute(0,2,3,1).squeeze(0).squeeze(-2)

# 从heatMap采样关键点
class KeypointSampler(nn.Module):
    '''
        Sample keypoints according to a Heatmap
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
    # 采样，可导?
    def sample(self, grid):
        '''
            Sample keypoints given a grid where each cell has logits stacked in last dimension
            Input
            grid: [B, C, H//w, W//w, w*w]

            Returns
            log_probs: [B, C, H//w, W//w ] - logprobs of selected samples
            choices: [B, C, H//w, W//w] indices of choices
            accept_mask: [B, C, H//w, W//w] mask of accepted keypoints
        '''
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
        idx_cells = self.gridify( 
            torch.dstack(torch.meshgrid(torch.arange(x.shape[-2], dtype=torch.float32),
                                        torch.arange(x.shape[-1], dtype=torch.float32),
                                                                      #indexing='ij')) \
                                                                                    )) \
            .permute(2,0,1).unsqueeze(0).expand(B,-1,-1,-1) ).to(x.device)

        log_probs, idx, mask = self.sample(keypoint_cells)
        keypoints = torch.gather(idx_cells, -1, idx.repeat(1,2,1,1).unsqueeze(-1)).squeeze(-1).permute(0,2,3,1)
    
        xy_probs = [  {'xy':keypoints[b][mask[b]].flip(-1), 'logprobs':log_probs[b][mask[b]]}
                    for b in range(B) ]
        return xy_probs                                                        

# 找到二次匹配一致的特征点，返回索引和相似度值        
class Matcher(nn.Module):
    '''
        Match two sets of features, and select mutual matches
        Input
            x: [M, D] features extracted from ref set
            y: [N, D] features extracted from dst set

        Returns
            log_probs:[t] - logprobs of selected matches
            dmatches: [t, 2] - (queryIdx, trainIdx) indices of selected matches
    '''  
    def __init__(self):
        super().__init__()
    
    def forward(self, x, y, T = 1.):
        Dmat = 2. - torch.cdist(x.unsqueeze(0), y.unsqueeze(0)).squeeze(0) # TODO 和文章中描述的不同
        
        logprob_rows = F.log_softmax(Dmat * T, dim=1)
        logprob_cols = F.log_softmax(Dmat.t() * T, dim=1)
        choice_rows = torch.argmax(logprob_rows, dim=1)
        choice_cols = torch.argmax(logprob_cols, dim=1)

        seq = torch.arange(choice_cols.shape[0], dtype = choice_cols.dtype, device = choice_cols.device)
        mutual = choice_rows[choice_cols] == seq

        logprob_rows = torch.gather(logprob_rows, -1, choice_rows.unsqueeze(-1)).squeeze(-1)
        logprob_cols = torch.gather(logprob_cols, -1, choice_cols.unsqueeze(-1)).squeeze(-1)

        log_probs = logprob_rows[choice_cols[mutual]] + logprob_cols[seq[mutual]]

        dmatches = torch.cat((choice_cols[mutual].unsqueeze(-1), seq[mutual].unsqueeze(-1)), dim=1)

        return  log_probs, dmatches

# 稠密匹配 TODO 有问题，感觉原理可以再优化，简单的加是不是不太合适
class DenseMatcher(nn.Module):
  '''
  Match all features and interpret them as raw logprobs
  Input
    x: [M, D] features extracted from ref set
    y: [N, D] features extracted from dst set

  Returns
    log_probs:[M, N] - logprobs of pairwise matches
  '''  
  def __init__(self): 
    super().__init__()

  def forward(self, x, y, T = 1.):
    Dmat = 2. - torch.cdist(x, y) 
    logprob_rows = F.log_softmax(Dmat * T, dim=1)
    logprob_cols = F.log_softmax(Dmat * T, dim=0)

    return logprob_rows + logprob_cols


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
# 描述子
class HardNet(nn.Module):
  def __init__(self, nchannels=3, out_ch = 128):
    super().__init__()

    self.nchannels = nchannels

    self.features = nn.Sequential(
      nn.InstanceNorm2d(self.nchannels),
      Pad2D(pad=(0,0,1,1), mode = 'circular'),
      nn.Conv2d(self.nchannels, 32, 3, bias=False, padding=(0,1)),
      nn.BatchNorm2d(32, affine=False),
      nn.ReLU(),
      Pad2D(pad=(0,0,1,1), mode = 'circular'),
      nn.Conv2d(32, 32, 3, bias=False, padding=(0,1)),
      nn.BatchNorm2d(32, affine=False),
      nn.ReLU(),
      Pad2D(pad=(0,0,1,1), mode = 'circular'),
      nn.Conv2d(32, 64, 3, bias=False, padding=(0,1), stride=2),
      nn.BatchNorm2d(64, affine=False),
      nn.ReLU(),
      Pad2D(pad=(0,0,1,1), mode = 'circular'),
      nn.Conv2d(64, 64, 3, bias=False, padding=(0,1)),
      nn.BatchNorm2d(64, affine=False),
      nn.ReLU(),
      Pad2D(pad=(0,0,1,1), mode = 'circular'),
      nn.Conv2d(64, 64, 3, bias=False, padding=(0,1), stride=2),
      nn.BatchNorm2d(64, affine=False),
      nn.ReLU(),
      Pad2D(pad=(0,0,1,1), mode = 'circular'),
      nn.Conv2d(64, 64, 3, bias=False, padding=(0,1)),
      nn.BatchNorm2d(64, affine=False),
      nn.ReLU(),
      nn.Dropout(0.1),

      #Vanilla Block
      # nn.Conv2d(128, 128, kernel_size=8, bias=False),
      # nn.BatchNorm2d(128, affine=False),

      #Rotation invariance block - pool angle axis
      nn.AvgPool2d((8,1), stride=1),
      nn.Conv2d(64, out_ch, (1,3), bias=False),
      nn.BatchNorm2d(out_ch, affine=False),
      nn.ReLU(),
      nn.Conv2d(out_ch, out_ch, (1,3), bias=False),
      nn.BatchNorm2d(out_ch, affine=False),
      nn.ReLU(),
      nn.Conv2d(out_ch, out_ch, (1,3), bias=False),
      nn.BatchNorm2d(out_ch, affine=False),
      nn.ReLU(),
      nn.Conv2d(out_ch, out_ch, (1,2), bias=False),
      nn.BatchNorm2d(out_ch, affine=False)
    )
    
  def forward(self, x):
    if x is not None:
      x = self.features(x).squeeze(-1).squeeze(-1)
      #x = F.normalize(x)
    return x


class SmallFCN(nn.Module):
  def __init__(self, nchannels=3):
    super().__init__()

    self.nchannels = nchannels

    self.features = nn.Sequential(
      nn.InstanceNorm2d(self.nchannels),
      Pad2D(pad=(0,0,1,1), mode = 'circular'),
      nn.Conv2d(self.nchannels, 8, 3, bias=False, padding=(0,1)),
      nn.BatchNorm2d(8, affine=False),
      nn.ReLU(),

      Pad2D(pad=(0,0,1,1), mode = 'circular'),
      nn.Conv2d(8, 16, 3, bias=False, padding=(0,1), stride=2),
      nn.BatchNorm2d(16, affine=False),
      nn.ReLU(),

      Pad2D(pad=(0,0,1,1), mode = 'circular'),
      nn.Conv2d(16, 16, 3, bias=False, padding=(0,1), stride=2),
      nn.BatchNorm2d(16, affine=False),
      nn.ReLU(),
      nn.Dropout(0.1),

      #Vanilla Block
      # nn.Conv2d(16, 32, kernel_size=8, bias=False),
      # nn.BatchNorm2d(32, affine=False),      

      #Rotation invariance block - pool angle axis
      nn.AvgPool2d((8,1), stride=1),
      nn.Conv2d(16, 32, (1,8), bias=False),
      nn.BatchNorm2d(32, affine=False),
    )
    
  def forward(self, x):
    x = self.features(x).squeeze(-1).squeeze(-1)
    x = F.normalize(x)
    return x


# ******************************** 网络TPS ********************************
# TODO 
class ThinPlateNet(nn.Module):
    def __init__(self, in_channels, n_channels=1, ctrlpts=(8,8), fixed_tps=False):
        super().__init__()
        self.ctrlpts = ctrlpts
        self.nctrl = ctrlpts[0]*ctrlpts[1]
        self.nparam = (self.nctrl+2)
        self.interpolator = InterpolateSparse2d(mode='bilinear')
        self.fixed_tps = fixed_tps

        self.fcn = nn.Sequential( 
                            nn.Conv2d(in_channels, in_channels*2, 3, padding = 1, stride=1, bias = False),
                            nn.BatchNorm2d(in_channels*2, affine=False),
                            nn.ReLU(),
                            nn.Conv2d(in_channels*2, in_channels*2, 3, bias = False),
                            nn.BatchNorm2d(in_channels*2, affine=False),
                            nn.ReLU())
        self.attn = nn.Sequential(
                                    nn.Linear(in_channels*2, in_channels*4),
                                    nn.BatchNorm1d(in_channels*4, affine = False),
                                    nn.ReLU(),
                                    nn.Dropout(0.1),
                                    nn.Linear(in_channels*4, in_channels*4),
                                    #nn.BatchNorm1d(in_channels*4, affine = False),
                                    nn.ReLU(),
                                    nn.Dropout(0.1),
                                    nn.Linear(in_channels*4, self.nparam*2),
                                    nn.Tanh())
        # 用于将特定层的参数初始化为零，以实现初始的恒等（identity）TPS（Thin Plate Splines）变换
        for i in [-2, -5, -9]:
            self.attn[i].weight.data.normal_(0., 1e-5)
            self.attn[i].bias.data.zero_() #normal_(0., 1e-5)

    # 用于获取以每个二维关键点位置为中心的极坐标网格。它计算出极坐标网格，然后将其转换为极坐标中的半径和角度。
    def get_polar_grid(self, keypts, Hs, Ws, coords='linear', gridSize=(32,32), maxR=32.):
        '''
            gets polar grids centered at each of 2D keypoint positions 
            Input:
                keypts [N,2] - 2D keypoints
                Hs, Ws (height & width of original img)

            Output:
                polargrid [N, h, w, 2] - polar grid for the N keypoints
        '''


    def forward(self, x, in_imgs, keypts, Ho, Wo):
        '''
            Input:
                x [B, C, H, W] - mid-level feature tensor from last layer of U-Net encoder
                in_imgs [B, C, H, W] - Original input images
                keypts [B, N, 2] - keypoints for each img in the batch
                Ho, Wo, original image Height & Width

            Output:
                patches List [[N, C, 32, 32]] - deformed polar patches of size (32, 32) (default HardNet patch size)
        '''
        patches = []
        B, C, _, _ = x.shape
        Theta = self.fcn(x) #compute TPS params




# ******************************** 网络DEAL ********************************
# ******************************** 网络DEAL ********************************





# ******************************** 网络DEAL ********************************
class DEAL(nn.Module):
    '''
        Base class for extracting deformation-aware keypoints and descriptors
        Input
            x: [B, C, H, W] Images

        Returns
            kpts:
            list of size [B] with detected keypoints
            descs:
            list of size [B] with descriptors 
    ''' 
    def __init__(self, enc_channels=[1,32,64,128], fixed_tps=False, mode=None):
        super().__init__()
        self.net = UNet(enc_channels) # 主干网络，用于特征编码和解码
        self.detector =KeypointSampler() # 用于从热图中采样关键点
        self.interpolator = InterpolateSparse2d() # 用于对稀疏二维数据进行插值

        hn_out_ch = 128 if mode == 'end2end-tps' else 64
        print('backbone: %d hardnet: %d'%(enc_channels[-1], hn_out_ch))
        # 用于学习可变形的极坐标网格
        self.tps_net = ThinPlateNet(in_channels=enc_channels[-1], n_channels=enc_channels[0], fixed_tps=fixed_tps)
        self.hardnet = HardNet(nchannels=enc_channels[0], out_ch=hn_out_ch)

        self.nchannels = enc_channels[0]
        self.enc_channels = enc_channels
        self.mode = mode
        if self.mode == 'ts-fl':
            print('adding fusion layer...')
            self.fusion_layer = nn.Sequential(nn.Linear(128, 128), nn.ReLU(),
                                              nn.Linear(128, 128), nn.Sigmoid())
    
    def subpix_refine(self, score_map, xy, size=3):
        # TODO 
        pass

    # 非极大值抑制
    def NMS(self, x, threshold=3., kernel_size=3):
        local_max = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size//2)(x)
        #local_min = -nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size//2)(-x)
        #pos = (x == local_max & ~(x == local_min)) & (x > threshold)
        pos = (x==local_max)&(x>threshold)
        return pos.nonzero()[..., 1:].flip(-1)  # 非零位置的索引，并通过flip(-1)将其从(y,x)变换为(x,y)
    
    # 根据特征点位置采样描述子
    def sample_descs(self, feature_map, kpts, H, W):
        # contiguous(),通常用于确保张量在内存中是连续存储的，而不是间隔存储的
        return self.interpolator(feature_map, kpts, H, W).contiguous()

    def forward(self, x, NMS=False, threshold=3., return_tensors=False, top_k=None):
        # 保证维度
        if self.nchannels == 1 and x.shape[1]!=1:
            x = torch.mean(x, axis=1, keepdim=True)
        
        B, C, H, W = x.shape
        out = self.net(x)
        #print(out['map'].shape, out['descr'].shape)
        if not NMS:
            kpts = self.detector(out['map'])
        else:
            kpts = [{'xy': self.NMS(out['map'][b], threshold)} for b in range(B)]
        
        # 训练期间对关键点(xy位置、概率)进行边界过滤
        if NMS:
            for b in range(B):
                filter_A = kpts[b]['xy'][:,0] >= 16 
                filter_B = kpts[b]['xy'][:,1] >= 16 
                filter_C = kpts[b]['xy'][:,0] < W - 16 
                filter_D = kpts[b]['xy'][:,1] < H - 16
                filter_all = filter_A * filter_B * filter_C * filter_D

                kpts[b]['xy'] = kpts[b]['xy'][filter_all]
                if 'logprobs' in kpts[b]:
                    kpts[b]['logprobs'] = kpts[b]['logprobs'][filter_all]
                # kpts[0]['xy'] = kpts[0]['xy'][:2,:]
                # kpts[0]['logprobs'] = kpts[0]['logprobs'][:2]
                    
        # 选top-k个关键点(根据热力图score)
        if top_k is not None:
            for b in range(B):
                scores = out['map'][b].squeeze(0)[kpts[b]['xy'][:,1].long(), kpts[b]['xy'][:,0].long()]
                sorted_idx = torch.argsort(-scores)
                kpts[b]['xy'] = kpts[b]['xy'][sorted_idx[:top_k]]
                if 'logprobs' in kpts[b]:
                    kpts[b]['logprobs'] = kpts[b]['xy'][sorted_idx[:top_k]]













        
class DALF_extractor:
    def __init__(self, model_path=None, 
                 device=torch.device('cpu'), fixed_tps=False):
        self.device = device
        print("running DALF model on ", self.device)

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
        

        



