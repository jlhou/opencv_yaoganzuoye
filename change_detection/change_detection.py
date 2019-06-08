import numpy as np
import cv2
import matplotlib.pyplot as plt


def Find_Threshold(delta):#OTSU寻找阈值
    val = np.zeros([256])
    for th in range(256):
        loc1 = delta > th
        loc2 = delta <= th
        '''delta[loc1]=255
        delta[loc2]=0'''
        if delta[loc1].size == 0:
            mu1 = 0
            omega1 = 0
        else:
            mu1 = np.mean(delta[loc1])
            omega1 = delta[loc1].size/delta.size

        if delta[loc2].size == 0:
            mu2 = 0
            omega2 = 0
        else:
            mu2 = np.mean(delta[loc2])
            omega2 = delta[loc2].size/delta.size
        val[th] = omega1*omega2*np.power((mu1-mu2),2)
    plt.figure()
    loc = np.where(val == np.max(val))
    plt.plot(val)
    plt.ylabel("Var")
    plt.xlabel("Threshold")
    plt.grid("on")

    print("\nThe best OTSU Threshold: ",loc[0])
    return loc[0]


def CD_diff(img1, img2):#影像差值法

    delta = np.subtract(img2, img1)
    sh = delta.shape
    delta += np.abs(delta.min())
    th = Find_Threshold(delta)
    if np.size(th) > 1:
        th = th[0]
    for i1 in range(sh[0]):
        for i2 in range(sh[1]):
            if delta[i1][i2] >= th:
                delta[i1][i2] = 0
            else:
                delta[i1][i2] = 255
    print(delta)
    cv2.imshow('delta', delta)
    cv2.imwrite('delta_change_detection.jpg', delta)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img1 = cv2.imread('1.jpg')
img2 = cv2.imread('2.jpg')
Grayimg1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
Grayimg2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
CD_diff(Grayimg1, Grayimg2)
