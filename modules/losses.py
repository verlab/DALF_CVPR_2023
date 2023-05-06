import torch
import kornia



def dual_hardnet_loss(X,Y, margin = 0.5, anchorSwap = False, random = False, mask = None):

  loss12 = hardnet_loss(X, Y, margin, anchorSwap, random, mask)
  loss21 = hardnet_loss(Y, X, margin, anchorSwap, random, mask)

  return (loss12 + loss21) / 2.


def hardnet_loss(X,Y, margin = 0.5, anchorSwap = False, random = False, mask = None):

  if X.size() != Y.size() or X.dim() != 2 or Y.dim() != 2:
    raise RuntimeError('Error: X and Y shapes must match and be 2D matrices')

  #dist_mat = torch.sqrt( 2.*(1.-torch.mm(X,Y.t())) )
  dist_mat = torch.cdist(X, Y, p=2.0)
  dist_pos = torch.diag(dist_mat)
  dist_neg = dist_mat + 100.*torch.eye(*dist_mat.size(), dtype = dist_mat.dtype, 
          device = dist_mat.get_device() if dist_mat.is_cuda else torch.device("cpu"))

  #filter repeated patches on negative distances to avoid weird stuff on gradients
  dist_neg = dist_neg + dist_neg.le(0.01).float()*100.

  #filter out negative samples that are on the same image according to mask (1D array with starting & ending
  # idx of each image in the descriptors array X and Y)
  if mask is not None:
    mask_filter = torch.zeros_like(dist_neg)
    for i in range(len(mask)-1):
      mask_filter[mask[i]:mask[i+1], mask[i]:mask[i+1]] = 100.
    dist_neg = dist_neg + mask_filter

  if random: # Triplet Loss on random negatives
    dist_neg = torch.sort(dist_neg)[0][:, 4:-8]
    rows = torch.arange(dist_neg.size(0), device = dist_neg.device) 
    cols = torch.randint(dist_neg.size(1), size = (dist_neg.size(0),), device = dist_neg.device)
    dist_neg = dist_neg[rows, cols]
    loss = torch.clamp(margin + dist_pos - dist_neg, min=0.)
  else: #Margin Ranking Loss
    hard_neg = torch.min(dist_neg, 1)[0]
    if anchorSwap:
      hard_neg = torch.min(hard_neg, torch.min(dist_neg, 0)[0])

    loss = torch.clamp(margin + dist_pos - hard_neg, min=0.)

  return loss.mean()

def SSIMLoss(p1, p2):
  return kornia.losses.ssim_loss(p1.mean(1, keepdim = True), p2.mean(1, keepdim = True), window_size = 7)

def sharpness_loss(img):
  sharpness = kornia.filters.laplacian(img,3).std()
  return torch.clamp(0.1 - sharpness, min = 0.) * 2.

def regularized_SSIM_loss(p1, p2):
  '''
     Avoids degenerate case of constant intensity patches by penalizing by patch sharpness
  '''
  ssiml = SSIMLoss(p1, p2)
  sharpnessl = sharpness_loss(p1) + sharpness_loss(p2)
  return (ssiml + sharpnessl) / 2.
