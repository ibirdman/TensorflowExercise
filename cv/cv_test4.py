import cv2
import numpy as np

img = cv2.imread("data/splitshape.png", cv2.IMREAD_GRAYSCALE)

rows, cols = img.shape
sobel_horizontal = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5) #利用sobel算子进行边缘检测，x方向
sobel_vertical = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5) #利用sobel算子进行边缘检测，y方向
kernel_1 = np.array([[-1,0,1], [-2,0,2], [-1,0,1]])   #利用图像卷积的方法进行边缘放大
kernel_2 = np.array([[-1,-2,-1], [0,0,0], [1,2,1]])    #利用图像卷积的方法进行边缘放大
output=cv2.filter2D(img, -1,kernel=kernel_1)
cv2.imshow('ker1',output)
output2=cv2.filter2D(img, -1,kernel=kernel_2)
cv2.imshow('ker2',output2)
cv2.imshow('Original', img)
cv2.imshow('Sobel horizontal', sobel_horizontal)
cv2.imshow('Sobel vertical', sobel_vertical)
cv2.waitKey(0)