from argparse import ArgumentParser

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot, ret_result
from mmseg.core.evaluation import get_palette
from PIL import Image
import py360convert
import numpy as np
import os

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
    img = np.array(Image.open(args.img))
    if len(img.shape) == 2:
        img = img[..., None]
    height, width, _ = img.shape

    # scroll image
    im_scroll = np.roll(img, width // 8 * 3, axis=1)
    img = py360convert.e2c(im_scroll, face_w=width//4, mode='bilinear')
    img = Image.fromarray(img)
    #img.show()
    im_path = os.getcwd() + "/demo/Image/CG_cube_single.png"
    img.save(im_path)
    result = inference_segmentor(model, im_path)
    # save result
    img = ret_result(model, args.img, result, get_palette(args.palette))
    #
    if len(img.shape) == 2:
        img = img[..., None]
    out = py360convert.c2e(img, h=height, w=width, mode='bilinear')
    out = np.roll(out, -width // 8 * 3, axis=1)
    out = out.astype(np.uint8).squeeze(2)
    out = Image.fromarray(out, 'L')
    out.show()
    out.save(args.outdir)


if __name__ == '__main__':
    main()
