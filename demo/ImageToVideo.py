import glob

import cv2

img_array = []
for filename in sorted(glob.glob("/home/hori/hdd/EgoPose/OpticalFlow/1013_take_008_of/*.png")):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

name = './Video/008_of.mp4'
out = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*'MP4V'), 30.0, size)

for i in range(len(img_array)):
    out.write(img_array[i])
    print(i)
out.release()