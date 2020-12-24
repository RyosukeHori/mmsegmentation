import cv2
import numpy as np
from PIL import Image

for j in range(0, 9165):
    # test a single image
    im1 = cv2.imread('./demo/Wrist_009_Mask_90/{:06d}.png'.format(j))
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2 = cv2.imread('./demo/Wrist_009_normal/{:06d}.png'.format(j))
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    OR_im = cv2.bitwise_or(im1, im2)

    cv2.imwrite('./demo/Wrist_009_normal&90/{:06d}.png'.format(j), OR_im)
    print('{:06d}.png saved'.format(j))