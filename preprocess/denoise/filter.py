#-*- coding:utf-8 -*-
"""
    desc: 滤波器操作,去除图像噪声
    author:MeteorMan
    datetime:2020/10/27
"""

import cv2
import numpy as np

img = cv2.imread('../img/lena_noise.png')

#平滑线性滤波
img_mean = cv2.blur(img, ksize=(5, 5))
cv2.imwrite("mean.png", img_mean)

# 高斯滤波
img_Guassian = cv2.GaussianBlur(img, (5, 5), 0)
cv2.imwrite("Guassian.png", img_Guassian)

# 中值滤波
img_median = cv2.medianBlur(img, 5)
cv2.imwrite("median.png", img_median)
# 双边滤波
img_bilater = cv2.bilateralFilter(img, 9, 75, 75)
cv2.imwrite("bilater.png", img_bilater)
