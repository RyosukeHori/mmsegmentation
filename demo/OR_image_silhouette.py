import cv2
import numpy as np
from PIL import Image

for j in range(0, 9164):
    # test a single image
    im1 = cv2.imread('./demo/1013_take_009_equilib2/{:06d}.png'.format(j))
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2 = cv2.imread('./demo/1013_take_009_equilib2/{:06d}.png'.format(j+1))
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    OR_im = cv2.bitwise_or(im1, im2)

    cv2.imwrite('./demo/1013_take_009_equilib2_interp/{:06d}.png'.format(j), OR_im)
    print('{:06d}.png saved'.format(j))