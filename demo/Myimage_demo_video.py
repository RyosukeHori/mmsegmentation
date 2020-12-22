from argparse import ArgumentParser

import numpy as np
import py360convert
import os
import cv2

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot, ret_result
from mmseg.core.evaluation import get_palette
from PIL import Image
import mmcv


def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='cityscapes',
        help='Color palette used for segmentation map')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device=args.device)

    res_width, res_height = 224, 224
    for j in range(0, 9165):
        # test a single image
        img_path = './demo/Wrist_1013_take_009/{:06d}.png'.format(j)
        orig_img = np.array(Image.open(img_path))
        if len(orig_img.shape) == 2:
            orig_img = orig_img[..., None]
        height, width, _ = orig_img.shape

        AND_im = np.zeros((height, width))
        # AND_im += 255
        AND_im = AND_im.astype(np.uint8)
        # scroll by 45 deg
        for i in range(360 // 90):
            im_scroll = np.roll(orig_img, width // 8 * i, axis=1)
            img = py360convert.e2c(im_scroll, face_w=width // 4, mode='bilinear')
            img = Image.fromarray(img)
            im_path = os.getcwd() + "/demo/Image/tmp.png"
            img.save(im_path)
            # segmentation
            result = inference_segmentor(model, im_path)
            img = ret_result(model, args.img, result, get_palette(args.palette))
            if len(img.shape) == 2:
                img = img[..., None]
            img_e = py360convert.c2e(img, h=height, w=width, mode='bilinear')
            img_e = np.roll(img_e, -width // 8 * i, axis=1).astype(np.uint8).squeeze(2)

            # and operation
            AND_im = cv2.bitwise_or(AND_im, img_e)
            # cv2.imshow("and", AND_im)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        # save result
        out = Image.fromarray(AND_im)
        #img.show()
        out2 = out.resize((res_width, res_height))
        out2.save('./demo/Wrist_009_Mask/{:06d}.png'.format(j))
        #img2.save('./demo/1_e2c_col.png'.format(i))
        print('{:06d}.png saved'.format(j))

if __name__ == '__main__':
    main()
