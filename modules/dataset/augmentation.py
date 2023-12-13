'''
    This script implements color + geometric transformations using Kornia
    Given a dataset of random images, we apply color transformations, 
    homography warps and also TPS warps.

    @author: Guilherme Potje

'''

import torch
from torch import nn
from torch.utils.data import Dataset
import torch.utils.data as data
from torchvision import transforms
import torch.nn.functional as F

import cv2
import kornia
import kornia.augmentation as K
from kornia.geometry.transform import get_tps_transform as findTPS
from kornia.geometry.transform import warp_points_tps, warp_image_tps

import glob
import random
import tqdm

import numpy as np
import pdb
import time

random.seed(0)
torch.manual_seed(0)

# 生成 Thin-Plate Spline (TPS) 变换所需的参数
def generateRandomTPS(shape, grid = (8, 6), GLOBAL_MULTIPLIER = 0.3, prob = 0.5):
    # 初始化网格和控制点：
    h, w = shape
    sh, sw = h/grid[0], w/grid[1]
    src = torch.dstack(torch.meshgrid(torch.arange(0, h + sh , sh),
                         torch.arange(0, w + sw , sw)))
    # 偏移控制点：
    offsets = torch.rand(grid[0]+1, grid[1]+1, 2) - 0.5
    offsets *= torch.tensor([ sh/2, sw/2 ]).view(1, 1, 2)  * min(0.97, 3. * GLOBAL_MULTIPLIER)
    dst = src + offsets if np.random.uniform() < prob else src # 生成目标点：
    # 归一化坐标系：
    src, dst = src.view(1, -1, 2), dst.view(1, -1, 2)
    src = (src / torch.tensor([h,w]).view(1,1,2) ) * 2 - 1.
    dst = (dst / torch.tensor([h,w]).view(1,1,2) ) * 2 - 1.
    weights, A = findTPS(dst, src) # 计算权重和 A 矩阵：

    return src, weights, A


# 生成随机的仿射变换矩阵
def generateRandomHomography(shape, GLOBAL_MULTIPLIER = 0.3):

    theta = np.radians(np.random.normal(0, 12.0*GLOBAL_MULTIPLIER)) # 生成随机旋转角度

    scale = np.random.normal(0, 1.1*GLOBAL_MULTIPLIER)#0.15*GLOBAL_MULTIPLIER) # 随机尺度
    if scale < 0.0: # get the right part of the gaussian
        scale = 1.0/(1. + abs(scale))
    else:
        scale = 1. + scale

    tx , ty = -shape[1]/2.0 , -shape[0]/2.0 
    txn, tyn = np.random.normal(0, 80.0*GLOBAL_MULTIPLIER, 2) #translation error 平移误差
    c, s = np.cos(theta), np.sin(theta)

    sx , sy = np.random.normal(0,0.6*GLOBAL_MULTIPLIER,2)
    p1 , p2 = np.random.normal(0,0.006*GLOBAL_MULTIPLIER,2)
    if np.random.uniform() > 0.5:
        scale_r = np.random.uniform(max(0.2, 0.8 - GLOBAL_MULTIPLIER), 1.)
    else:
        scale_r = np.random.uniform(1., min(4., 1.2 + 3.*GLOBAL_MULTIPLIER))

    H_t = np.array(((1,0, tx), (0, 1, ty), (0,0,1))) #t

    H_r = np.array(((c,-s, 0), (s, c, 0), (0,0,1))) #rotation,
    H_a = np.array(((1,sy, 0), (sx, 1, 0), (0,0,1))) # affine
    H_p = np.array(((1, 0, 0), (0 , 1, 0), (p1,p2,1))) # projective

    H_s = np.array(((scale,0, 0), (0, scale * scale_r, 0), (0,0,1))) #scale
    H_b = np.array(((1.0,0,-tx +txn), (0, 1, -ty + tyn), (0,0,1))) #t_back,

    #H = H_e * H_s * H_a * H_p
    H = np.dot(np.dot(np.dot(np.dot(np.dot(H_b,H_s),H_p),H_a),H_r),H_t)

    return H

# 数据增强
class AugmentationPipe(nn.Module):
    def __init__(
                self, device, load_dataset = True,
                img_dir = "/homeLocal/guipotje/sfm_datasets/downloads/*/images/*.jpg",
                warp_resolution = (1200, 900),
                out_resolution = (400, 300),
                max_num_imgs = 40,
                num_test_imgs = 200,
                batch_size = 6,
                ):
        super(AugmentationPipe, self).__init__()
        self.half = 16
        self.device = device
        self.sample_img = cv2.imread('./assets/kanagawa_1.png') # 图像样例
        self.dims = warp_resolution
        self.batch_size = batch_size
        self.out_resolution = out_resolution 
        self.dims_t = torch.tensor([int(self.dims[0]*0.8) - int(self.dims[0]*0.2) -1,
                                    int(self.dims[1]*0.8) - int(self.dims[1]*0.2) -1]).float().to(device).view(1,1,2)
        self.dims_s = torch.tensor([ self.dims_t[0,0,0] / out_resolution[0],
                                     self.dims_t[0,0,1] / out_resolution[1]]).float().to(device).view(1,1,2) 
        self.sample_img = cv2.resize(self.sample_img, self.dims)
        self.all_imgs = glob.glob(img_dir)
        random.shuffle(self.all_imgs)

        # 组合图像增强操作
        self.aug_list = kornia.augmentation.ImageSequential(
                        kornia.augmentation.RandomChannelShuffle(p=0.5), # 随机重排图像的通道顺序，重排概率为50%
                        kornia.augmentation.ColorJitter(0.2, 0.2, 0.2, 0.2, p=1.), # 进行颜色抖动，增加图像的对比度、亮度和色调。(对比度、亮度、饱和度、色相的变化范围)
                        kornia.augmentation.RandomEqualize(p = 0.5), # 对图像进行直方图均衡化，提高图像的对比度，均衡化的概率为50%。
                        kornia.augmentation.RandomGaussianBlur(p = 0.3, sigma = (2.5, 2.5), kernel_size = (7,7)) # 进行随机高斯模糊处理。(高斯核)
                        )
        
        # 调整裁剪并通过特征点选择满足要求的图片
        if load_dataset:
            print('Found a total of ', len(self.all_imgs), ' images for training..')

            if len(self.all_imgs) - num_test_imgs < max_num_imgs:
                raise RuntimeError('Error: test set overlaps with training set! Decrease number of test imgs')

            train = []
            fast = cv2.FastFeatureDetector_create(30)
            for p in tqdm.tqdm(self.all_imgs[:max_num_imgs], desc='loading train'): # 训练数据集(裁剪、大小调整、满足图像特征点)
                im = cv2.imread(p)
                halfH, halfW = im.shape[0]//2, im.shape[1]//2
                if halfH > halfW:
                    im = np.rot90(im)
                    halfH, halfW = halfW, halfH
                im = im[halfH-self.dims[1]//2:halfH+self.dims[1]//2, halfW-self.dims[0]//2:halfW+self.dims[0]//2, :]
                #print (im.shape)
                if im.shape[0] != self.dims[1] or im.shape[1] != self.dims[0]:
                    #print('resizing..')
                    im = cv2.resize(im, self.dims)
                if len(fast.detect(im)) > 1_000:
                    train.append(np.copy(im))

            self.train = train
            
            self.test = [
                        cv2.resize(cv2.imread(p), self.dims)                 
                        for p in tqdm.tqdm(self.all_imgs[-num_test_imgs:],
                                           desc='loading test')
                        ] 

            self.TPS = True

    # 将输入的点集进行归一化处理，将坐标映射到 [-1, 1] 的范围内
    def norm_pts_grid(self, x):
        if len(x.size()) == 2:
            return (x.view(1,-1,2) * self.dims_s / self.dims_t) * 2. - 1 
        return (x * self.dims_s / self.dims_t) * 2. - 1

    # 对归一化后的点集进行反归一化操作，将其从 [-1, 1] 的范围映射回原始的坐标空间
    def denorm_pts_grid(self, x):
        if len(x.size()) == 2:
            return ((x.view(1,-1,2) + 1) / 2.) / self.dims_s * self.dims_t
        return ((x+1) / 2.) / self.dims_s * self.dims_t

    # 生成随机的关键点
    def rnd_kps(self, shape, n = 256):
        h, w = shape
        kps = torch.rand(size = (3,n)).to(self.device) # [0,1)范围，之后调整到图像范围
        kps[0,:]*=w
        kps[1,:]*=h
        kps[2,:] = 1.0

        return kps

    # 对输入的点进行仿射变换
    def warp_points(self, H, pts):
      scale = self.dims_s.view(-1,2)
      offset = torch.tensor([int(self.dims[0]*0.2), int(self.dims[1]*0.2)], device = pts.device).float()
      pts = pts*scale + offset # 缩放和偏移
      pts = torch.vstack( [pts.t(), torch.ones(1, pts.shape[0], device = pts.device)])
      warped = torch.matmul(H, pts)
      warped = warped / warped[2,...] 
      warped = warped.t()[:, :2]
      return (warped - offset) / scale

    def forward(self, x, difficulty = 0.3, TPS = False, prob_deformation = 0.5, test = False):
        with torch.no_grad(): # 不需要梯度
            x = (x/255.).to(self.device) # [0,1]范围
            shape = x.shape[-2:]
            h, w = shape

            #t0 = time.time()
            output = self.aug_list(x) # 数据增强

            #Correlated Gaussian Noise 随机选择是否添加相关的高斯噪声，并在需要时添加到增强后的图像中。
            if np.random.uniform() > 0.5: # 50%的概率
                noise = F.interpolate(torch.randn_like(output)*(16/255), (h//2, w//2))
                noise = F.interpolate(noise, (h, w), mode = 'bicubic')
                output = torch.clip( output + noise, 0., 1.)

            #print('done in ', time.time() - t0)
            # 生成随机仿射变换矩阵
            H = torch.tensor([generateRandomHomography(shape, difficulty) for b in range(self.batch_size)],
                               dtype = torch.float32).to(self.device)
            # 对图像进行仿射变换，可以实现平移、旋转、缩放等操作
            output = kornia.geometry.transform.warp_perspective(output, H,
                            dsize = shape, padding_mode = 'zeros')


            #crop 20% of image boundaries each side to reduce invalid pixels after warps
            # 裁剪图像边界来减少变换后可能产生的无效像素
            low_h = int(h * 0.2); low_w = int(w*0.2)
            high_h = int(h*0.8); high_w= int(w * 0.8)
            output = output[..., low_h:high_h, low_w:high_w]

            #apply TPS if desired:
            # 进行 Thin-Plate Spline (TPS) 变换
            if TPS:
                src, weights, A = None, None, None
                for b in range(self.batch_size):
                    b_src, b_weights, b_A = generateRandomTPS(shape, (8,6), difficulty, prob = prob_deformation)
                    b_src, b_weights, b_A = b_src.to(self.device), b_weights.to(self.device), b_A.to(self.device)

                    if src is None:
                        src, weights, A = b_src, b_weights, b_A
                    else:
                        src = torch.cat((b_src, src))
                        weights = torch.cat((b_weights, weights))
                        A = torch.cat((b_A, A))

                #print(output.shape, src.shape, weights.shape, A.shape)
                output = warp_image_tps(output, src, weights, A) # TPS 变换被用于对输出的图像进行形变，从而产生不同的变换效果。

            output = F.interpolate(output, self.out_resolution[::-1], mode = 'bilinear', align_corners = False) # 插值

        
        if TPS:
            return output, (H, src, weights, A)
        else:
            return output, H

    # 计算目标关键点在变换后的图像中的对应位置
    def get_correspondences(self, kps_target, T):
        H, H2, src, W, A = T
        undeformed  = self.denorm_pts_grid(   
                                        warp_points_tps(self.norm_pts_grid(kps_target),
                                        src, W, A) ).view(-1,2)

        warped_to_src = self.warp_points(H@torch.inverse(H2), undeformed)

        return warped_to_src