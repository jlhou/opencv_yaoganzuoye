import cv2
img = cv2.imread('2.jpg', 0)
ret, th1 = cv2.threshold(img, 75, 255, cv2.THRESH_BINARY)
cv2.imshow('th1', th1)
cv2.waitKey(0)
cv2.imwrite('water_extract.jpg', th1)
cv2.destroyAllWindows()
