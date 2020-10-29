#-*- coding:utf-8 -*-
"""
    desc: 腐蚀、膨胀等相关运算
    author:MeteorMan
    datetime:2020/10/27
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

#读取图像
img = cv2.imread('img/book.png', 0)

# 使用 getStructuringElement 定义结构元素，shape 为结构元素的形状，0:矩形；1:十字交
# 叉形；2:椭圆形；ksize 为结构元素的大小；anchor 为原点的位置，默认值（-1，-1）表示原点
# 为结构元素的中心点
k = cv2.getStructuringElement(shape=1, ksize=(3, 3), anchor=(-1, -1))
# k = np.ones((3,3),np.uint8) 也可以自己定义一个结构元素

# erode 函数实现腐蚀运算，src 为输入图像，kernel 为之前定义的结构元素，iterations 为
# 腐蚀操作次数
erosion = cv2.erode(src=img, kernel=k, iterations=1)
cv2.imwrite("Eroded_Image.png", erosion)

dilation = cv2.dilate(img, k, iterations=1)
cv2.imwrite("Dilated_Image.png", dilation)

# morphologyEx 函数实现开闭运算, src 为输入图像，op 为运算类型，cv2.MORPH_OPEN：开
# 运算；cv2.MORPH_CLOSE：闭运算，kernel 为结构元素
opening = cv2.morphologyEx(src=img, op=cv2.MORPH_OPEN, kernel=k)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel=k)
cv2.imwrite("Opening_Image.png", opening)
cv2.imwrite("Closing_Image.png", closing)
