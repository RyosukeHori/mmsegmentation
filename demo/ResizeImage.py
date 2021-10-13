from PIL import Image
import cv2

width, height = 224, 224
for i in range(0, 1390):

    img = cv2.imread('/home/hori/Downloads/0121_take/{:06d}.png'.format(i))
    #im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (width, height))
    #th, img_gray_th = cv2.threshold(im_gray, 128, 255, cv2.THRESH_OTSU)
    #cv2.imwrite('./demo/before_resized/1013_take_008_resized/{:06d}.png'.format(i), img_gray_th)
    cv2.imwrite('/home/hori/Downloads/0121_take_RGB/{:06d}.png'.format(i), img)
    #img = cv2.imread('./demo/before_resized/1013_take_008_resized/{:06d}.png'.format(i))
    print('{:06d}.png saved'.format(i))