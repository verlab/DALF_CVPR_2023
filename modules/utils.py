from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
from kornia.geometry.transform import warp_points_tps, warp_image_tps
import io

grad_fig = None


def grab_mpl_fig(fig):
    '''
        Transform current drawn fig into a np array
    '''
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='raw', dpi=100)
    io_buf.seek(0)
    img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                     newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    io_buf.close()
    return img_arr
    #plt.imshow(img_arr) ; plt.show(); input()

def plot_grid(warped, title = 'Grid Vis', mpl = True):
    #visualize 
    g = None
    n = warped[0].shape[0]

    for i in range(0, n, 16):
        if i + 16 <= n:
            for w in warped:
                pad_val = 0.7 if i//16%2 == 1 else 0
                gw = make_grid(w[i:i+16].detach().clone().cpu(), padding=4, pad_value=pad_val, nrow=16)
                g = gw if g is None else torch.cat((g, gw), 1)

    if mpl:
        fig = plt.figure(figsize = (12, 3), dpi=100)
        plt.imshow(np.clip(g.permute(1,2,0).numpy()[...,::-1], 0, 1))
        return fig


def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []

    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n) and p.grad is not None:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu().numpy())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation=60, ha = 'right')
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.tight_layout()


def make_batch_sfm(augmentor, difficulty = 0.3, train = True):
    Hs = []
    img_list = augmentor.train if train else augmentor.test
    dev = augmentor.device
    batch_images = []

    with torch.no_grad(): # we dont require grads in the augmentation
        for b in range(augmentor.batch_size):
            rdidx = np.random.randint(len(img_list))
            img = torch.tensor(img_list[rdidx], dtype=torch.float32).permute(2,0,1).to(augmentor.device).unsqueeze(0)
            batch_images.append(img)

        batch_images = torch.cat(batch_images)

        p1, H1 = augmentor(batch_images, difficulty)
        p2, H2 = augmentor(batch_images, difficulty, TPS = True, prob_deformation = 0.7)

        H2, src, W, A = H2

        for b in range(augmentor.batch_size):
            Hs.append((H2[b]@torch.inverse(H1[b]).to(dev), 
                        src[b].unsqueeze(0),
                        W[b].unsqueeze(0),
                        A[b].unsqueeze(0)))

    return p1, p2, Hs



def get_reward(kps1, kps2, H, augmentor, penalty = 0., px_thr = 2):
    with torch.no_grad():
        #perspective transform 2 -> 1
        if not augmentor.TPS:
            warped = augmentor.warp_points(torch.inverse(H), kps2)
        else:
        #first undeform TPS, then perspective 2 -> 1
            H, src, W, A = H
            undeformed  = augmentor.denorm_pts_grid(   \
                                          warp_points_tps(augmentor.norm_pts_grid(kps2),
                                          src, W, A) ).view(-1,2)
            warped = augmentor.warp_points(torch.inverse(H), undeformed)
            
        error = torch.linalg.norm(warped - kps1, dim = 1)
        rewards = (error <= px_thr).float()
        reward_sum = torch.sum(rewards)
        rewards[rewards == 0.] = penalty
    return rewards, reward_sum

def get_dense_rewards(kps1, kps2, H, augmentor, penalty = 0., px_thr = 1.5):
    with torch.no_grad():
        #perspective transform 2 -> 1
        if not augmentor.TPS:
            warped = augmentor.warp_points(torch.inverse(H), kps2)
        else:
        #first undeform TPS, then perspective 2 -> 1
            H, src, W, A = H
            undeformed = augmentor.denorm_pts_grid(   \
                                          warp_points_tps(augmentor.norm_pts_grid(kps2),
                                          src, W, A) ).view(-1,2)
            warped = augmentor.warp_points(torch.inverse(H), undeformed)
            
        d_mat = torch.cdist(kps1, warped)
        x_vmins, x_mins = torch.min(d_mat, dim=1)
        y_mins = torch.arange(len(x_mins)).long()

        d_mat[y_mins, x_mins] *= -1.
        d_mat[d_mat >= 0.] = 0.
        d_mat[d_mat < -px_thr] = 0.
        d_mat[d_mat != 0.] = 1.

        reward_mat = d_mat
        reward_sum = reward_mat.sum() 
        reward_mat[reward_mat == 0.] = penalty
    return reward_mat, reward_sum
    
    
def get_positive_corrs(kps1, kps2, H, augmentor, i=0, px_thr = 1.5):
    with torch.no_grad():
        #perspective transform 2 -> 1
        if not augmentor.TPS:
            warped = augmentor.warp_points(torch.inverse(H), kps2['xy'])
        else:
        #first undeform TPS, then perspective 2 -> 1
            H, src, W, A = H
            undeformed = augmentor.denorm_pts_grid(   \
                                          warp_points_tps(augmentor.norm_pts_grid(kps2['xy']),
                                          src, W, A) ).view(-1,2)
            warped = augmentor.warp_points(torch.inverse(H), undeformed)
              
        d_mat = torch.cdist(kps1['xy'], warped)
        x_vmins, x_mins = torch.min(d_mat, dim=1)
        y_mins = torch.arange(len(x_mins), device= d_mat.device).long()

        #grab indices of positive correspodences & filter too close kps in the same image
        y_mins = y_mins[(x_vmins < px_thr)] #* (self_vmins > 2.)]
        x_mins = x_mins[(x_vmins < px_thr)] #* (self_vmins > 2.)]

    return torch.hstack((y_mins.unsqueeze(1),x_mins.unsqueeze(1))),  \
           kps1['patches'][y_mins], kps2['patches'][x_mins]
  