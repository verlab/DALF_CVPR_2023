import numpy as np
import torch

def normalize(pts):
    pts = pts - pts.mean()
    norm_avg = (pts**2).sum(axis=1).sqrt().mean()
    pts = pts / norm_avg * np.sqrt(2.)
    return pts

def random_choice(max_idx, batch, dev):
    return torch.randint(max_idx, (batch, 3), device = dev)

def nr_RANSAC(ref_pts, tgt_pts, device, batch = 4_000, thr = 0.1):
    """
    Computes non-rigid RANSAC.

    Args:
        ref_pts (numpy.ndarray): A numpy array of shape (N, 2) containing the reference points.
        tgt_pts (numpy.ndarray): A numpy array of shape (N, 2) containing the target points.
        device (torch.device): The device to use for computations.
        batch (int, optional): The batch size to use. Defaults to 4000.
        thr (float, optional): The threshold value for inliers. Defaults to 0.1.

    Returns:
        numpy.ndarray: An array of indices of shape (M,) representing the inlier matches.

    Raises:
        ValueError: If ref_pts or tgt_pts have incorrect shapes.

    Notes:
        This function computes non-rigid RANSAC to find a transformation between two sets of matched points.
        The algorithm uses PyTorch to compute a set of random choices and a hypothesis, and then evaluates
        the inliers for each hypothesis. The hypothesis with the most inliers is returned as the best estimate
        for the transformation. The function returns an array of indices representing the inlier matches.
    """
    
    ref_pts = torch.tensor(ref_pts)
    tgt_pts = torch.tensor(tgt_pts)
    with torch.no_grad():
        ref_pts = ref_pts.to(device)
        tgt_pts = tgt_pts.to(device)

        ref_pts = normalize(ref_pts)
        tgt_pts = normalize(tgt_pts)
        pts = torch.cat((ref_pts, tgt_pts), axis=1)
        choices = random_choice(len(pts), batch, dev = device)
        batched_pts = pts[choices]
        batched_pts = batched_pts.permute(0,2,1)
        mean_vec = batched_pts.mean(axis=2)

        batched_pts = batched_pts - mean_vec.view(-1,4,1)

        U, S, Vh = torch.linalg.svd(batched_pts)
        A = U[:, :, :2]

        #check if hypothesis is not ill-conditioned (has 2 sing vals > eps)
        good_mask = S[:, 1] > 1e-3

        pts_expanded = pts.expand(batch,-1,-1).permute(0,2,1)
        M = torch.bmm(A, A.permute(0,2,1))
        residuals = pts_expanded - torch.bmm( torch.bmm(A, A.permute(0,2,1)) , (pts_expanded - mean_vec.view(-1,4,1)) )  - mean_vec.view(-1,4,1)
        residuals = torch.linalg.norm(residuals, dim=1)

        inliers = residuals < thr
        inliers = inliers[good_mask]
        count = inliers.sum(dim=1)
        best = count.argmax()

    return inliers[best].cpu().numpy()

