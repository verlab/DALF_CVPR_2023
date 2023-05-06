from torch.utils.tensorboard import SummaryWriter
import torch
import os

def check_dir(f):
	if not os.path.exists(f):
		os.makedirs(f)

class TrainLogger():
    def __init__(self, logdir, name):
        self.logdir = logdir + '/' + name
        check_dir(self.logdir)
        self.writer = SummaryWriter(self.logdir)

    def log_scalars(self, step, avg_det, acc, inliers, kp_rewards, hard_loss, ssim_loss):
        self.writer.add_scalar('All/ #Det Keypoints', avg_det, step)
        self.writer.add_scalar('All/ Match Acc.', acc, step)
        self.writer.add_scalar('All/ #Inliers', inliers, step)
        self.writer.add_scalar('All/ #Rewards (keypoints)', kp_rewards, step)
        self.writer.add_scalar('All/ Hard Loss', hard_loss, step)
        self.writer.add_scalar('All/ SSIM Loss (1 to 0)', ssim_loss, step)

    def log_fig(self, step, fig, name):
        fig_tensor = torch.tensor(fig).permute(2,0,1)
        self.writer.add_image('Plot from step ' + str(step)+'/'+name, fig_tensor)



