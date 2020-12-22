from argparse import ArgumentParser

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot, ret_result
from mmseg.core.evaluation import get_palette
from PIL import Image
import py360convert
import numpy as np
import os
import cv2

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
    parser.add_argument('--outdir', help='Output Dir')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    '''
    # test a single image
    result = inference_segmentor(model, args.img)
    # show the results
    show_result_pyplot(model, args.img, result, get_palette(args.palette))
    '''
    orig_img = np.array(Image.open(args.img))
    if len(orig_img.shape) == 2:
        orig_img = orig_img[..., None]
    height, width, _ = orig_img.shape

    AND_im = np.zeros((height, width))
    #AND_im += 255
    AND_im = AND_im.astype(np.uint8)
    # scroll by 45 deg
    for i in range(360 // 45):
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
        #cv2.imshow("and", AND_im)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

    out = Image.fromarray(AND_im, 'L')
    out.show()
    out.save(args.outdir)


if __name__ == '__main__':
    main()
