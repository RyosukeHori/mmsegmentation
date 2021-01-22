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
    for j in range(74, 9166):
        # test a single image
        img_path = './demo/1013_take_009/{:06d}.png'.format(j)
        orig_img = np.array(Image.open(img_path))

        if len(orig_img.shape) == 2:
            orig_img = orig_img[..., None]
        height, width, _ = orig_img.shape

        # normal segmentation image as OR Image
        im_gray = ret_result(model, img_path, inference_segmentor(model, img_path), get_palette(args.palette))
        im_gray = im_gray.astype(np.uint8)

        OR_im = np.zeros(im_gray.shape).astype(im_gray.dtype)
        retval, im_th = cv2.threshold(im_gray, 127, 255, cv2.THRESH_OTSU)
        contours, hierarchy = cv2.findContours(im_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            areas = [cv2.contourArea(c) for c in contours]
            max_index = np.argmax(areas)
            cv2.drawContours(OR_im, contours, max_index, 255, -1)

        # scroll by 45 deg
        for i in range(360 // 45):
            # scroll image
            im_scroll = np.roll(orig_img, width // 8 * i, axis=1)
            tmp = Image.fromarray(im_scroll)
            tmp.save("./demo/tmp/tmp0.png")

            # e2c
            img = py360convert.e2c(im_scroll, face_w=width // 4, mode='bilinear')
            img = Image.fromarray(img)
            im_path = os.getcwd() + "/demo/Image/tmp.png"
            img.save(im_path)
            img.save("./demo/tmp/tmp1.png")

            # segmentation
            result = inference_segmentor(model, im_path)
            img = ret_result(model, args.img, result, get_palette(args.palette))
            tmp = Image.fromarray(img)
            tmp.save("./demo/tmp/tmp2.png")
            if len(img.shape) == 2:
                img = img[..., None]

            # c2e
            img_e = py360convert.c2e(img, h=height, w=width, mode='bilinear')

            # scroll
            img_e = np.roll(img_e, -width // 8 * i, axis=1).astype(np.uint8).squeeze(2)
            tmp = Image.fromarray(img_e)
            tmp.save("./demo/tmp/tmp3.png")
            #cv2.imshow("img_e", img_e)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

            # extract contors
            external = np.zeros(im_gray.shape).astype(im_gray.dtype)
            retval, im_th = cv2.threshold(img_e, 127, 255, cv2.THRESH_OTSU)
            contours, hierarchy = cv2.findContours(im_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            areas = [cv2.contourArea(c) for c in contours]
            max_index = np.argmax(areas)
            cv2.drawContours(external, contours, max_index, 255, -1)
            tmp = Image.fromarray(external)
            tmp.save("./demo/tmp/tmp4.png")

            # OR operation
            OR_im = cv2.bitwise_or(OR_im, external)
            #cv2.imshow("and", OR_im)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
        external = OR_im

        if (any(OR_im[:, 0]) or any(OR_im[:, -1])) and not (all(OR_im[0]) or all(OR_im[-1])):
            external2 = external.copy()
            areas[max_index] = 0
            max_index = np.argmax(areas)
            cv2.drawContours(external2, contours, max_index, 255, -1)
            if any(external2[:, 0]) and any(external2[:, -1]):
                external = external2

        # save result
        out = Image.fromarray(external)
        #out.show()
        out2 = out.resize((res_width, res_height))
        out2.save('./demo/1013_take_009_cont/{:06d}.png'.format(j))
        #img2.save('./demo/1_e2c_col.png'.format(i))
        print('{:06d}.png saved'.format(j))

if __name__ == '__main__':
    main()
