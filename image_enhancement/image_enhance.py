import cv2
import numpy as np

img = cv2.imread("SAR.tif", 0)
cv2.imshow("src", img)
dst = cv2.equalizeHist(img)
cv2.imshow("dst", dst)

img_median = cv2.medianBlur(dst, 5)
cv2.imshow("img_median", img_median)

im_color = cv2.applyColorMap(img_median, cv2.COLORMAP_JET)
cv2.imshow("im_color", im_color)
cv2.imwrite('SAR_enhance.tif', im_color)
cv2.waitKey(0)
