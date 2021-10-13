from argparse import ArgumentParser

import numpy as np
import py360convert
import os
import cv2
import os.path as osp

from typing import Union

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot, ret_result
from mmseg.core.evaluation import get_palette
from PIL import Image
import mmcv
from equilib import Equi2Equi


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

    img_path = '/home/hori/Downloads/0121_take/000000.png'
    #img_path = './demo/1013_take_009/000000.png'
    orig_img = np.array(Image.open(img_path))
    height, width, _ = orig_img.shape

    # Initialize equi2equi
    equi2equi = Equi2Equi(
        w_out=width,
        h_out=height,
        sampling_method="default",
        mode="bilinear",
    )

    def preprocess(
            img: Union[np.ndarray, Image.Image],
            is_cv2: bool = False,
    ) -> np.ndarray:
        """Preprocesses image"""
        if isinstance(img, np.ndarray) and is_cv2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if isinstance(img, Image.Image):
            # Sometimes images are RGBA
            img = img.convert("RGB")
            img = np.asarray(img)
        assert len(img.shape) == 3, "input must be dim=3"
        assert img.shape[-1] == 3, "input must be HWC"
        img = np.transpose(img, (2, 0, 1))
        return img

    rot_45 = {
        "roll": 0,  #
        "pitch": -np.pi / 6,  # vertical
        "yaw": 0,  # horizontal
    }

    rot_minus45 = {
        "roll": 0,  #
        "pitch": np.pi / 6,  # vertical
        "yaw": 0,  # horizontal
    }

    for j in range(0, 1390):
        # test a single image
        img_path = '/home/hori/Downloads/0121_take/{:06d}.png'.format(j)
        #img_path = './demo/1013_take_009/{:06d}.png'.format(j)
        orig_img = np.array(Image.open(img_path))

        AND_im = np.zeros((height, width)).astype(np.uint8)

        # scroll by 180 deg
        num_rot = 360 // 180
        for i in range(num_rot):
            OR_im = np.zeros((height, width)).astype(np.uint8)
            # scroll image
            im_scroll = np.roll(orig_img, width // num_rot * i , axis=1)
            src_img = Image.fromarray(im_scroll)

            src_img = preprocess(src_img)

            # change pitch
            img_45 = equi2equi(
                src=src_img,
                rot=rot_45,
            )
            img_minus45 = equi2equi(
                src=src_img,
                rot=rot_minus45,
            )

            src_img = np.transpose(src_img, (1, 2, 0))
            src_img = Image.fromarray(src_img)
            img_45 = np.transpose(img_45, (1, 2, 0))
            img_45 = Image.fromarray(img_45)
            img_minus45 = np.transpose(img_minus45, (1, 2, 0))
            img_minus45 = Image.fromarray(img_minus45)

            src_img_path = os.getcwd() + "/demo/tmp/src_img-2.png"
            src_img.save(src_img_path)
            img_45_path = os.getcwd() + "/demo/tmp/img_45-2.png"
            img_45.save(img_45_path)
            img_minus45_path = os.getcwd() + "/demo/tmp/img_minus45-2.png"
            img_minus45.save(img_minus45_path)

            # segmentation
            result_src = inference_segmentor(model, src_img_path)
            result_45 = inference_segmentor(model, img_45_path)
            result_minus45 = inference_segmentor(model,img_minus45_path)
            src_img = ret_result(model, args.img, result_src, get_palette(args.palette))
            img_45 = ret_result(model, args.img, result_45, get_palette(args.palette))
            img_minus45 = ret_result(model, args.img, result_minus45, get_palette(args.palette))

            '''
            tmp_src_img = Image.fromarray(src_img)
            tmp_img_45 = Image.fromarray(img_45)
            tmp_img_minus45 = Image.fromarray(img_minus45)
            src_img_path = os.getcwd() + "/demo/tmp/src_img_seg-2.png"
            tmp_src_img.save(src_img_path)
            img_45_path = os.getcwd() + "/demo/tmp/img_45_seg-2.png"
            tmp_img_45.save(img_45_path)
            img_minus45_path = os.getcwd() + "/demo/tmp/img_minus45_seg-2.png"
            tmp_img_minus45.save(img_minus45_path)
            '''

            img_45 = np.stack((img_45,) * 3, -1)
            img_45 = preprocess(img_45)
            #img_45 = np.transpose(img_45, (2, 0, 1))
            img_minus45 = np.stack((img_minus45,) * 3, -1)
            img_minus45 = preprocess(img_minus45)
            #img_minus45 = np.transpose(img_minus45, (2, 0, 1))

            # change pitch
            img_45 = equi2equi(
                src=img_45,
                rot=rot_minus45,
            )
            img_minus45 = equi2equi(
                src=img_minus45,
                rot=rot_45,
            )

            img_45 = np.transpose(img_45, (1, 2, 0))
            img_minus45 = np.transpose(img_minus45, (1, 2, 0))

            '''
            tmp_src_img = Image.fromarray(src_img)
            tmp_img_45 = Image.fromarray(img_45)
            tmp_img_minus45 = Image.fromarray(img_minus45)
            src_img_path = os.getcwd() + "/demo/tmp/src_img_seg_ori.png"
            tmp_src_img.save(src_img_path)
            img_45_path = os.getcwd() + "/demo/tmp/img_45_seg_ori.png"
            tmp_img_45.save(img_45_path)
            img_minus45_path = os.getcwd() + "/demo/tmp/img_minus45_seg_ori.png"
            tmp_img_minus45.save(img_minus45_path)
            '''

            src_img = np.roll(src_img, -width // num_rot * i, axis=1).astype(np.uint8)
            img_45 = np.roll(img_45, -width // num_rot * i, axis=1).astype(np.uint8)[:, :, 0]
            img_minus45 = np.roll(img_minus45, -width // num_rot * i, axis=1).astype(np.uint8)[:,:,0]

            OR_im = cv2.bitwise_or(OR_im, src_img)
            OR_im = cv2.bitwise_or(OR_im, img_45)
            OR_im = cv2.bitwise_or(OR_im, img_minus45)

            '''
            ORim_path = os.getcwd() + "/demo/tmp/OR_im.png"
            out = Image.fromarray(OR_im)
            out.save(ORim_path)
            '''

            external = np.zeros(OR_im.shape).astype(OR_im.dtype)
            retval, im_th = cv2.threshold(OR_im, 127, 255, cv2.THRESH_OTSU)
            contours, hierarchy = cv2.findContours(im_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                areas = [cv2.contourArea(c) for c in contours]
                max_index = np.argmax(areas)
                cv2.drawContours(external, contours, max_index, 255, -1)
                OR_im = external

            if any(OR_im[:, 0]) or any(OR_im[:, -1]):
                external2 = OR_im.copy()
                areas[max_index] = 0
                max_index = np.argmax(areas)
                cv2.drawContours(external2, contours, max_index, 255, -1)
                if any(external2[:, 0]) and any(external2[:, -1]):
                    OR_im = external2

            '''
            ORim_path = os.getcwd() + "/demo/tmp/OR_im2.png"
            out = Image.fromarray(OR_im)
            out.save(ORim_path)
            '''

            # AND operation
            if i is 0:
                AND_im = cv2.bitwise_or(AND_im, OR_im)
            else:
                AND_im = cv2.bitwise_and(AND_im, OR_im)

            '''
            ANDim_path = os.getcwd() + "/demo/:tmp/AND_im.png"
            out = Image.fromarray(AND_im)
            out.save(ANDim_path)
            '''

        # save result
        out = Image.fromarray(AND_im)
        out.save('./demo/tmp/extract.png')
        #out.show()
        out2 = out.resize((res_width, res_height))
        out2.save('./demo/0121_take_equilib2/{:06d}.png'.format(j))
        #out2.save('./demo/1013_take_009_equilib2/{:06d}.png'.format(j))
        #img2.save('./demo/1_e2c_col.png'.format(i))
        print('{:06d}.png saved'.format(j))

if __name__ == '__main__':
    main()
