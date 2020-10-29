#-*- coding:utf-8 -*-
"""
    desc: 二值化算法：OSTU
    author:MeteorMan
    datetime:2020/10/27
"""

import cv2
from matplotlib import pyplot as plt

#读取图片
img = cv2.imread('../img/2-1.png')

#1.将图像转换为灰度图
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#绘制灰度图
plt.subplot(311)#采用3行1列布局，本次占用第1行第1列
plt.imshow(gray_img, 'gray')
plt.title('input image')
plt.xticks([])
plt.yticks([])

#2.对灰度图使用ostu算法
ret1, th1 = cv2.threshold(gray_img, 0, 255, cv2.THRESH_OTSU)
#绘制灰度直方图
plt.subplot(312)
plt.hist(gray_img.ravel(), 256)
#标注ostu阈值所在直线
plt.axvline(x=ret1, color='red', label='ostu')
plt.legend(loc='upper right')
plt.title('Histogram')
plt.xticks([])
plt.yticks([])

#绘制二值化图像
plt.subplot(313)
plt.imshow(th1, "gray")
plt.title('output image')
plt.xticks([])
plt.yticks([])
plt.show()

