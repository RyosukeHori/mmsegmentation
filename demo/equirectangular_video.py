from argparse import ArgumentParser

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

    width, height = 224, 224
    for i in range(0, 9924):
        # test a single image
        img_path = './demo/1013_take_002/{:06d}.png'.format(i)
        #img_path = './demo/1_e2c.png'.format(i)
        result = inference_segmentor(model, img_path)

        # save result
        img = Image.fromarray(ret_result(model, img_path, result, get_palette(args.palette)))
        #img.show()
        img2 = img.resize((width, height))
        img2.save('./demo/1013_take_002_equi/{:06d}.png'.format(i))
        #img2.save('./demo/1_e2c_col.png'.format(i))
        print('{:06d}.png saved'.format(i))

if __name__ == '__main__':
    main()
