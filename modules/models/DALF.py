'''
  - This script contains the model definitions for detection of kps trained with policy gradient;
  - Description is performed by the deformation-aware network and optimized with hard triplet loss;
  - Detection and description are simulateously optimized to be deformation-aware.
'''

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import math
import pdb, tqdm
import numpy as np
import cv2

import time
import os

import torchvision.transforms as transforms

from modules.tps import pytorch as TPS
#from modules import utils

class DALF_extractor:
    """
    Class for extracting local features (keypoints and descriptors) using the DALF method.

    Args:
        model (str, optional): Path to the pre-trained DALF model file. Defaults to None, in which case the default
            pre-trained model shipped with the library will be used.
        dev (torch.device, optional): Device on which to run the model. Defaults to `torch.device('cpu')`.
        fixed_tps (bool, optional): Whether to use fixed identity TPS (polar grid) during feature extraction. Defaults to False.

    Attributes:
        net (DEAL): The DALF network model used for feature extraction.

    Raises:
        RuntimeError: If the network mode cannot be parsed from the model file name.
    """

    def __init__(self, model=None, 
                       dev = torch.device('cpu'),
                       fixed_tps = False):
        self.dev = dev

        print('running DALF on', self.dev)

        if model is None:
          abs_path = os.path.dirname(os.path.abspath(__file__))
          model = abs_path + '/../../weights/model_ts-fl_final.pth'

        if 'end2end-backbone' in model:
          backbone_nfeats = 128
        else:
          backbone_nfeats = 64
        
        modes = ['end2end-backbone', 'end2end-tps', 'end2end-full', 'ts1', 'ts2', 'ts-fl']
        
        mode = None
        for m in modes:
          if m in model:
            mode = m

        if mode is None:
          raise RuntimeError('Could not parse network mode from file name - it has to be present')

        self.net = DEAL(enc_channels = [1, 32, 64, backbone_nfeats], fixed_tps = fixed_tps, mode = mode).to(dev)
        self.net.load_state_dict(torch.load(model, map_location=dev))

        self.net.eval().to(dev)
    

    def detectAndCompute(self, og_img, mask = None, top_k = 2048, return_map = False, threshold = 25., MS = False):
        """
        Detects and computes deformation-aware local features (keypoints and descriptors) from an input image.

        Args:
            og_img (numpy.ndarray or str): Input image, either as a NumPy array or path to the image file.
            mask (optional): Unused argument.
            top_k (int, optional): Maximum number of keypoints to return. Defaults to 2048.
            return_map (bool, optional): Whether to return a heatmap of the detected keypoints. Defaults to False.
            threshold (float, optional): Threshold used to filter out low-scoring keypoints. Defaults to 25.
            MS (bool, optional): Whether to perform multi-scale feature detection. Defaults to False

        Returns:
            tuple: A tuple of two or three elements, depending on the value of the `return_map` parameter. The first element is
            a list of `cv2.KeyPoint` objects representing the detected keypoints. The second element is a NumPy array of shape
            `(N, 128)` containing the corresponding descriptors, where `N` is the number of detected keypoints. If `return_map`
            is True, the third element is a NumPy array of the same shape as the input image, representing a heatmap of the
            detected keypoints.

        Raises:
            RuntimeError: If the input image cannot be loaded.

        """

        t0 = time.time()
        if MS:
          scales = [1/6, 1/4, 1/2, 1]
        else:
          scales = [1]

        kpts_list, descs_list, scores_list = [], [], []
        hd_map = None

        if isinstance(og_img, str):
            og_img = cv2.cvtColor(cv2.imread(og_img), cv2.COLOR_BGR2RGB)
            if og_img is None:
                raise RuntimeError('Image couldnt be loaded')

        for scale in scales:
          with torch.no_grad():
              img = cv2.resize(og_img, None, fx = scale, fy = scale, interpolation = cv2.INTER_AREA) if scale != 1. else og_img
              img = torch.tensor(img, dtype = torch.float32, device=self.dev).permute(2,0,1).unsqueeze(0)/255.

              kpts, descs, fmap = self.net(img, NMS = True, threshold = threshold, return_tensors = True, top_k = top_k)
              score_map = fmap['map'][0].squeeze(0).cpu().numpy()
              #utils.plot_grid([kpts[0]['patches'][:16]])
              kpts, descs = kpts[0]['xy'].cpu().numpy().astype(np.int16), descs[0].cpu().numpy()
              scores = score_map[kpts[:,1], kpts[:,0]]
              scores /= score_map.max()

              sort_idx = np.argsort(-scores)
              kpts, descs, scores = kpts[sort_idx], descs[sort_idx], scores[sort_idx]

              if return_map and hd_map is None:
                max_val = float(score_map.max())
                for p in kpts.astype(np.int32):
                  if False:#score_map[p[1],p[0]] > threshold:
                      cv2.drawMarker(score_map, (p[0], p[1]) , max_val, cv2.MARKER_CROSS, 6, 2)#, line_type = cv2.LINE_AA)
                hd_map = score_map

              #rescale kps
              kpts = kpts / scale

              kpts_list.append(kpts)
              descs_list.append(descs)
              scores_list.append(scores)
              

        # all_kpts = np.vstack(kpts_list)
        # all_descs = np.vstack(descs_list)
        # all_scores = np.hstack(scores_list)
        # sort_idx = np.argsort(-scores)
        # all_kpts, all_descs, all_scores = all_kpts   [sort_idx][:top_k], \
        #                                   all_descs  [sort_idx][:top_k], \
        #                                   all_scores [sort_idx][:top_k]


        #Try to balance keypoints between scales      
        if len(scales) > 1:
          perscale_kpts = top_k // len(scales)
          all_kpts = np.vstack([kpts[:perscale_kpts] for kpts in kpts_list[:-1]])
          all_descs = np.vstack([descs[:perscale_kpts] for descs in descs_list[:-1]])
          all_scores = np.hstack([scores[:perscale_kpts] for scores in scores_list[:-1]])

          all_kpts = np.vstack([all_kpts, kpts_list[-1][:(top_k - len(all_kpts))]])
          all_descs = np.vstack([all_descs, descs_list[-1][:(top_k - len(all_descs))]])
          all_scores = np.hstack([all_scores, scores_list[-1][:(top_k - len(all_scores))]])
        else:
          all_kpts = kpts_list[0]; all_descs = descs_list[0]; all_scores = scores_list[0]

        cv_kps = [cv2.KeyPoint(all_kpts[i][0], all_kpts[i][1], 6, 0, all_scores[i]) for i in range(len(all_kpts))]
        # print('took %.3f s'%(time.time() - t0))
        if return_map:
          return cv_kps, all_descs, hd_map
        else:
          return cv_kps, all_descs

    def detect(self, img, _ = None):
        return self.detectAndCompute(img)[0]


# 稀疏的 2D 位置信息在 3D 张量上进行插值。 TODO
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

  def normgrid(self, x, H, W):
    return 2. * (x/(torch.tensor([W-1, H-1], device = x.device, dtype = x.dtype))) - 1.

  def forward(self, x, pos, H, W):
    grid = self.normgrid(pos, H, W).unsqueeze(0).unsqueeze(-2) 

    x = F.grid_sample(x.unsqueeze(0), grid, mode = self.mode, align_corners = True)
    return x.permute(0,2,3,1).squeeze(0).squeeze(-2)

# 根据热图（Heatmap）采样关键点（keypoints）TODO
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
  def __init__(self, window_size = 8): 
    super().__init__()
    self.window_size = window_size

  # 将输入的热图张量进行划分，划分成一个个的窗口。
  def gridify(self, x):
    B, C, H, W = x.shape
    x = x.unfold(2, self.window_size, self.window_size)                              \
          .unfold(3, self.window_size, self.window_size)                             \
          .reshape(B, C, H//self.window_size, W//self.window_size, self.window_size**2)

    return x

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
    idx_cells = self.gridify( torch.dstack(torch.meshgrid(torch.arange(x.shape[-2], dtype=torch.float32),
                                                          torch.arange(x.shape[-1], dtype=torch.float32),
                                                          #indexing='ij'))     \
                                                                        ))     \
                                                         .permute(2,0,1).unsqueeze(0) 
                                                         .expand(B,-1,-1,-1) ).to(x.device)
                                                         

    log_probs, idx, mask = self.sample(keypoint_cells)

    keypoints = torch.gather(idx_cells, -1, idx.repeat(1,2,1,1).unsqueeze(-1)).squeeze(-1).permute(0,2,3,1)
    
    xy_probs = [  {'xy':keypoints[b][mask[b]].flip(-1), 'logprobs':log_probs[b][mask[b]]}
                  for b in range(B) ]

    return xy_probs


class Matcher(nn.Module):
  '''
  Match two sets of features, and select mutual matches
  Input
    x: [M, D] features extracted from ref set
    y: [N, D] features extracted from dst set

  Returns
    log_probs:[N] - logprobs of selected matches
    dmatches: [N, 2] - (queryIdx, trainIdx) indices of selected matches
  '''  
  def __init__(self): 
    super().__init__()

    
  def forward(self, x, y, T = 1.):
    Dmat = 2. - torch.cdist(x.unsqueeze(0), y.unsqueeze(0)).squeeze(0)
    
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
  def __init__(self, enc_channels = [1, 32, 64, 128], fixed_tps = False, mode = None): 
    super().__init__()
    self.net = UNet(enc_channels)
    self.detector = KeypointSampler()
    self.interpolator = InterpolateSparse2d()
   
    hn_out_ch = 128 if mode == 'end2end-tps' else 64 

    print('backbone: %d hardnet: %d'%(enc_channels[-1], hn_out_ch))
    self.tps_net = ThinPlateNet(in_channels = enc_channels[-1], nchannels = enc_channels[0],
                                 fixed_tps = fixed_tps)
    self.hardnet =  HardNet(nchannels = enc_channels[0], out_ch = hn_out_ch)

    self.nchannels = enc_channels[0]
    self.enc_channels = enc_channels
    self.mode = mode
    if self.mode == 'ts-fl':
      print('adding fusion layer...')
      self.fusion_layer = nn.Sequential(nn.Linear(128, 128), nn.ReLU(),
                                        nn.Linear(128, 128), nn.Sigmoid())


  def subpix_refine(self, score_map, xy, size = 3):
    '''
    apparently this function makes scores worse
    '''
    from kornia.geometry.subpix import dsnt
    if size%2 == 0:
      raise RuntimeError('Grid size must be odd')
    
    _, H, W = score_map.shape
    score_map = score_map.unsqueeze(1).expand(xy.shape[0], -1, -1, -1)

    #build a size x size grid around each keypoint to compute a local patch score for each keypoint
    g = torch.arange(size) - size//2
    gy, gx = torch.meshgrid(g, g)
    center_grid = torch.cat([gx.unsqueeze(-1), gy.unsqueeze(-1)], -1).to(xy.device)
    grids = center_grid.unsqueeze(0).repeat(xy.shape[0], 1, 1, 1) 
    grids = (grids + xy.view(-1, 1, 1, 2)) / torch.tensor([W-1, H-1]).to(xy.device)
    grids = grids * 2 - 1
    patches_scores = F.grid_sample(score_map, grids, mode='nearest', align_corners=True)

    #compute 2d expectation over local patches
    patches_scores = F.softmax(patches_scores.view(-1, size*size)/1., dim=-1).view(-1, 1, size, size)
    xy_offsets = dsnt.spatial_expectation2d(patches_scores, False).view(-1, 2) - size//2
    #print(xy_offsets[:4])
    xy = xy.float() + xy_offsets

    return xy

  # 非最大值抑制
  def NMS(self, x, threshold = 3., kernel_size = 3):
    pad=kernel_size//2
    local_max = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=pad)(x)
    #local_min = -nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=pad)(-x)
    #pos = (x == local_max & ~(x == local_min)) & (x > threshold)
    pos = (x == local_max) & (x > threshold)
    return pos.nonzero()[..., 1:].flip(-1)

  def sample_descs(self, feature_map, kpts, H, W):
    return self.interpolator(feature_map, kpts, H, W).contiguous()

  def forward(self, x, NMS = False, threshold = 3., return_tensors = False, top_k = None):

    if self.nchannels == 1 and x.shape[1] != 1:
      x = torch.mean(x, axis = 1, keepdim = True)

    B, C, H, W = x.shape
    out = self.net(x)
    #print(out['map'].shape, out['descr'].shape)
    if not NMS:
      kpts = self.detector(out['map'])
    else:
      kpts = [{'xy':self.NMS(out['map'][b], threshold)} for b in range(B)]

    #filter kps on border during training 训练期间对关键点进行边界过滤
    if not NMS:
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
    
    # 选top-k个关键点
    if top_k is not None:
      for b in range(B):
        scores = out['map'][b].squeeze(0)[kpts[b]['xy'][:,1].long(), kpts[b]['xy'][:,0].long()]
        sorted_idx = torch.argsort(-scores)
        kpts[b]['xy'] = kpts[b]['xy'][sorted_idx[:top_k]]
        if 'logprobs' in kpts[b]:
          kpts[b]['logprobs'] = kpts[b]['xy'][sorted_idx[:top_k]]

    #check if we need to optimize the tps module
    optimize_tps = self.mode == 'ts2' or self.mode == 'end2end-tps' or self.mode == 'end2end-full' or self.mode == 'ts-fl'
    #print('optimize tps? ', optimize_tps, self.mode)
    if optimize_tps:
      patches = self.tps_net(out['feat'], x, kpts, H, W)
    for b in range(B): 
      if optimize_tps:
        kpts[b]['patches'] = patches[b] 
      else:
        with torch.no_grad():
          kpts[b]['patches'] = torch.zeros(len(kpts[b]['xy']),1,32,32).to(x.device)

    if NMS:
      if len(kpts[b]['xy']) == 1:
        raise RuntimeError('No keypoints detected.')
        
      if self.mode == 'end2end-full' or self.mode == 'ts2' or self.mode == 'ts-fl':
        #distinct & invariant features : 64 + 64 dims
        if self.mode == 'ts-fl': #fuse descriptors with a MLP
          final_desc =  torch.cat((
            self.hardnet(patches[b]), 
            self.interpolator(out['feat'][b], kpts[b]['xy'], H, W)
            ), dim=1) 
          final_desc = self.fusion_layer(final_desc) * final_desc       
        else:
          final_desc = torch.cat((
            self.hardnet(patches[b]), 
            self.interpolator(out['feat'][b], kpts[b]['xy'], H, W)
            ), dim=1)
      elif self.mode == 'end2end-backbone' or self.mode == 'ts1':
        #full distinct features from backbone only: 128 dims
        final_desc = self.interpolator(out['feat'][b], kpts[b]['xy'], H, W)
      else:
        #full invariant: 128 dims
        final_desc = self.hardnet(patches[b])

      descs = [ F.normalize(final_desc) for b in range(B) ]

    if not NMS:
      if not return_tensors:
        return kpts
      else:
        return kpts, out 
    else:
      #for b, k in enumerate(kpts): 
      #  k['xy'] = self.subpix_refine(out['map'][b], k['xy'])
      if not return_tensors:
        return kpts, descs
      else:
        return kpts, descs, out

# 用于计算模型中可训练参数的数量
def get_nb_trainable_params(model):
	model_parameters = filter(lambda p: p.requires_grad, model.parameters())
	nb_params = sum([np.prod(p.size()) for p in model_parameters])
 
	print('Number of trainable parameters: {:d}'.format(nb_params))



class ThinPlateNet(nn.Module):
  '''
  Network definition to compute learnable deformable polar patches
  '''
  def __init__(self, in_channels, nchannels=1, ctrlpts=(8,8), fixed_tps=False):
    super().__init__()
    self.ctrlpts = ctrlpts
    self.nctrl = ctrlpts[0]*ctrlpts[1]
    self.nparam = (self.nctrl + 2)
    self.interpolator = InterpolateSparse2d(mode = 'bilinear')
    self.fixed_tps = fixed_tps

    self.fcn = nn.Sequential( 
                                nn.Conv2d(in_channels, in_channels*2, 3, padding = 1, stride=1, bias = False),
                                nn.BatchNorm2d(in_channels*2, affine=False),
                                nn.ReLU(),
                                nn.Conv2d(in_channels*2, in_channels*2, 3, bias = False),
                                nn.BatchNorm2d(in_channels*2, affine=False),
                                nn.ReLU(),
                              )

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
                                nn.Tanh(),
                              )
    #zero-out layer params for initial identity TPS transform
    for i in [-2, -5, -9]:
      self.attn[i].weight.data.normal_(0., 1e-5) 
      self.attn[i].bias.data.zero_()#normal_(0., 1e-5)


  def get_polar_grid(self, keypts, Hs, Ws, coords = 'linear', gridSize = (32,32), maxR = 32.):
    '''
    gets polar grids centered at each of 2D keypoint positions 
    Input:
          keypts [N,2] - 2D keypoints
          Hs, Ws (height & width of original img)

    Output:
          polargrid [N, h, w, 2] - polar grid for the N keypoints
    '''
    maxR = torch.ones_like(keypts[:,0]) * maxR
    self.batchSize = keypts.shape[0]
    
    ident = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], device = keypts.device).expand(self.batchSize, -1, -1)
    grid = F.affine_grid(ident, (self.batchSize, 1) + gridSize, align_corners= False)
    grid_y = grid[..., 0].view(self.batchSize , -1)
    grid_x = grid[..., 1].view(self.batchSize , -1)

    maxR = torch.unsqueeze(maxR, -1).expand(-1, grid_y.shape[-1]).float().to(keypts.device)

    # get radius of polar grid with values within [1, maxR]
    normGrid = (grid_y + 1) / 2
    if coords == "log": r_s_ = torch.exp(normGrid * torch.log(maxR))
    elif coords == "linear": r_s_ = 1 + normGrid * (maxR - 1)
    else: raise RuntimeError('Invalid coords type, choose [log, linear]')

    # convert radius values to [0, 2maxR/W] range
    r_s = (r_s_ - 1) / (maxR - 1) * 2 * maxR / Ws  


    # y is from -1 to 1; theta is from 0 to 2pi
    t_s = (
        grid_x + 1
    ) * np.pi

    x_coord = torch.unsqueeze(keypts[:, 0], -1).expand(-1, grid_x.shape[-1]) / Ws * 2. - 1.
    y_coord = torch.unsqueeze(keypts[:, 1], -1).expand(-1, grid_y.shape[-1]) / Hs * 2. - 1.

    aspectRatio = Ws/Hs

    x_s = r_s * torch.cos(
        t_s
    ) + x_coord 
    y_s = r_s * torch.sin(
        t_s
    ) * aspectRatio + y_coord


    polargrid = torch.cat(
          (x_s.view(self.batchSize , gridSize[0], gridSize[1], 1),
          y_s.view(self.batchSize , gridSize[0], gridSize[1], 1)),
          -1)
    
    return polargrid

  def forward(self, x, in_imgs, keypts, Ho ,Wo):
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

    for b in range(B):
      if keypts[b]['xy'] is not None and len(keypts[b]['xy']) >= 16:
        polargrid = self.get_polar_grid(keypts[b]['xy'], Ho, Wo)
        N, H, W, _ = polargrid.shape

        kfactor = 0.3
        offset = (1.0 - kfactor)/2.
        vmin = polargrid.view(N,-1,2).min(1)[0].unsqueeze(1).unsqueeze(1)
        vmax = polargrid.view(N,-1,2).max(1)[0].unsqueeze(1).unsqueeze(1)
        ptp = vmax - vmin
        polargrid = (polargrid - vmin) / ptp
        #scale by a factor and normalize to center for better condition
        polargrid = polargrid * kfactor + offset

        grid_img = polargrid.permute(0,3,1,2) # Trick to allow interpolating a torch 2D meshgrid into desired shape
        ctrl = F.interpolate(grid_img, self.ctrlpts).permute(0,2,3,1).view(N,-1,2)

        theta = self.interpolator(Theta[b], keypts[b]['xy'], Ho, Wo) #interpolate param tensor @ kp pos
        theta = self.attn(theta)
        theta = theta.view(-1, self.nparam, 2)
        #theta = theta.view(-1, 2, 8, 8)
        #theta = F.interpolate(theta, (32,32), mode = 'bicubic').permute(0,2,3,1)

        I_polargrid = theta.new(N, H, W, 3) #create a new tensor with identity polar grid (normalized by keypoint attributes)
        I_polargrid[..., 0] = 1.0
        I_polargrid[..., 1:] = polargrid

        if not self.fixed_tps:
          z = TPS.tps(theta, ctrl, I_polargrid)
          tps_warper = (I_polargrid[...,1:] + z) # *2-1
          #tps_warper = polargrid + theta
        else:
          tps_warper = polargrid

        #reverse transform - scale by a factor and normalize to center for better condition
        tps_warper = (tps_warper - offset) / kfactor        
        #denormalize each element in batch
        tps_warper = tps_warper * ptp + vmin

        curr_patches = F.grid_sample( in_imgs[b].expand(N,-1,-1,-1),
                                      tps_warper, align_corners = False,  padding_mode = 'zeros')
        
        patches.append(curr_patches)
      else:
        patches.append(None)
    return patches


# 填充方式
class Pad2D(torch.nn.Module):
  def __init__(self, pad, mode): 
    super().__init__()
    self.pad = pad
    self.mode = mode

  def forward(self, x):
    return F.pad(x, pad = self.pad, mode = self.mode)

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

################################### U-Net definition ####################################
#########################################################################################


class DownBlock(nn.Module):
  def __init__(self, in_ch, out_ch):
      super().__init__()
      self.convs = nn.Sequential( 
                                  #Gaussian2d(in_ch),
                                  nn.Conv2d(in_ch, out_ch, 3 , bias = False, padding = 1),
                                  nn.BatchNorm2d(out_ch, affine=False),
                                  nn.ReLU(),
                                  nn.Conv2d(out_ch, out_ch, 3, bias = False, padding = 1),
                                  nn.BatchNorm2d(out_ch, affine=False),
                                  nn.ReLU(), 
                                  nn.MaxPool2d(2)                                          
                                  )
  def forward(self, x):
    return self.convs(x)

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

# encoder: 每一层都保留
class Encoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.norm = nn.InstanceNorm2d(channels[0], affine=False)
        self.blocks = nn.ModuleList([DownBlock(channels[i], channels[i+1]) 
                                        for i in range(len(channels) -1)])

    def forward(self, x):

      x = self.norm(x)
      features = [x]

      for b in self.blocks:
        x = b(x)
        features.append(x)
      return features


class Decoder(nn.Module):
    def __init__(self, enc_ch, dec_ch):
        super().__init__()
        enc_ch = enc_ch[::-1]
        print(enc_ch)
        print(dec_ch)
        self.convs = nn.ModuleList( [UpBlock(enc_ch[i+1] + dec_ch[i], dec_ch[i+1]) 
                                                      for i in range(len(dec_ch) -2)] )
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
          upsampled = F.interpolate(x_next, size = x[i+1].size()[-2:], mode = 'bilinear', align_corners = True)
          x_next = torch.cat([upsampled, x[i+1]], dim = 1)
          #print(x_next.shape, '-->')
          x_next = self.convs[i](x_next)
          #print(x_next.shape)
        
        x_next = F.interpolate(x_next, size = x[-1].size()[-2:], mode = 'bicubic', align_corners = True)
        #print(x_next.shape)
        #print('-----------')
        return self.conv_heatmap(x_next)



class UNet(nn.Module):
    '''
    U-Net implementation for keypoint detection
    Input
      enc_channels: List of desired #n of intermediate feature map channels

    Returns
      dict:
        'map': heatmap of detected keypoints
        'descr': dense descriptors (spatially downsampled by a factor of approx. 8)
    '''  
    def __init__(self, enc_channels = [1, 32, 64, 128]):
      super().__init__()
      dec_channels = enc_channels[::-1] # 翻转
      self.encoder = Encoder(enc_channels)
      self.decoder = Decoder(enc_channels, dec_channels)
      self.nchannels = enc_channels[0]
      self.features = nn.Sequential( 
                                      nn.Conv2d(enc_channels[-1], enc_channels[-1], 3, padding = 1, bias = False),
                                      nn.BatchNorm2d(enc_channels[-1], affine=False),
                                      nn.ReLU(),                                                                                                                                                         
                                      nn.Conv2d(enc_channels[-1], enc_channels[-1], 1),
                                      nn.BatchNorm2d(enc_channels[-1], affine=False)
                                    )


    def forward(self, x):

      if self.nchannels == 1 and x.shape[1] != 1:
        x = torch.mean(x, axis = 1, keepdim = True)

      feats = self.encoder(x)
      out = self.decoder(feats)
      #feat = feats[-1]
      feat = self.features(feats[-1])

      return {'map':out, 'feat':feat}
      
 ################################################################################################################

