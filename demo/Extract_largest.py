import cv2
import numpy as np

im = cv2.imread("./demo/Image/tmp_OR.png")
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
external = np.zeros(im_gray.shape).astype(im_gray.dtype)
retval, im_bw = cv2.threshold(im_gray, 127, 255, cv2.THRESH_OTSU)

contours, hierarchy = cv2.findContours(im_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

areas = [cv2.contourArea(c) for c in contours]
max_index = np.argmax(areas)
cnt = contours[max_index]
cv2.drawContours(external, contours, max_index, 255, -1)

cv2.imshow("external", external)
cv2.waitKey(0)
cv2.destroyAllWindows()

