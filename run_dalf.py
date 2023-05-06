
'''
    Minimal snippet to extract DALF features from an image, following
    OpenCV's features2D interface standard.
'''

from modules.models.DALF import DALF_extractor as DALF
import torch
import cv2

dalf = DALF(dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

img = cv2.imread('./assets/kanagawa_1.png')

kps, descs = dalf.detectAndCompute(img)

print('--------------------------------------------------')
print("Number of Keypoints (list of cv2.KeyPoint): ", len(kps))
print("Descriptors (ndarray) shape: ", descs.shape)