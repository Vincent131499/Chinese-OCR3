# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 10:13:25 2019

@author: Ma Zhenwei
"""
import os
import json
import matplotlib.pyplot as plt
import matplotlib.cm as CM
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import cv2


def generate_region_score(img, shapes):
    '''
    img：维度(h, w, c)
    shapes：标注文件中的shapes
    '''
    h, w, c = img.shape
    region_map = np.zeros((h, w), dtype=float)
    
    for s in shapes:
        points = s['points'] # 4个点  
        # 将顺时针的标注 变成 逆时针 以符合cv2函数的要求
        points = [(int(points[0][0]),int(points[0][1])),(int(points[3][0]),int(points[3][1])),
             (int(points[2][0]),int(points[2][1])),(int(points[1][0]),int(points[1][1]))]
        
        dst = generate_transformed_gaussian_kernel(h, w, points)

        # 叠加到 region_map
        region_map += dst       
    return region_map
    

def generate_affinity_score(img, shapes):
    '''
    img：维度(h, w, c)
    shapes：标注文件中的shapes
    '''
    h, w, c = img.shape
    affinity_map = np.zeros((h, w), dtype=float)
    
    for i in range(len(shapes)-1):
        # 第一个字符位置 & 第二个字符位置
        points1 = np.float32(shapes[i]['points'])
        points2 = np.float32(shapes[i+1]['points'])
        # 第一个字符中心 & 第二个字符中心
        center1 = np.sum(np.array(points1),axis=0) / 4
        center2 = np.sum(np.array(points2),axis=0) / 4
        # 生成affinity box的4个顶点
        top_left = (points1[0] + points1[1] + center1) /3
        top_right = (points2[0] + points2[1] + center2) /3
        down_left = (points1[2] + points1[3] + center1) /3
        down_right = (points2[2] + points2[3] + center2) /3
        points = np.float32([top_left, down_left, down_right, top_right])
        dst = generate_transformed_gaussian_kernel(h, w, points)
        
        affinity_map += dst
    return affinity_map
        

def generate_transformed_gaussian_kernel(h, w, points):
    '''
    使用透视变换的高斯核建模region或affinity
    h：图像的高
    w：图像的宽
    points：维度(4,2)
    '''
    # 生成高斯核
    minX, minY = points[0]
    maxX, maxY = points[0]
    for i in range(1,4):
        minX = min(points[i][0],minX)
        minY = min(points[i][1],minY)
        maxX = max(points[i][0],maxX)
        maxY = max(points[i][1],maxY)
    kernel_w = int((maxX - minX + 1) // 2 * 2)
    kernel_h = int((maxY - minY + 1) // 2 * 2)
    
    kernel_size = 31
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size//2, kernel_size//2] = 1
    kernel = gaussian_filter(kernel, 10, mode='constant')
    
    kernel_size = max(kernel_h, kernel_w)
    kernel = cv2.resize(kernel,(kernel_size,kernel_size))
    
    # 将高斯核透视变换，坐标(列，行)
    src = np.float32([(0,0),(0,kernel_size),(kernel_size,kernel_size),(kernel_size,0)]) # 左上，左下，右下，右上
    tgt = np.float32(points)
    M = cv2.getPerspectiveTransform(src, tgt)
    dst = cv2.warpPerspective(kernel, M, (w,h))
    
    # 转换到[0.001,1]之间
    mini = dst[np.where(dst>0)].min()
    maxi = dst[np.where(dst>0)].max()
    h = 1
    l = 0.001 # 与预训练模型的分布保持一致
    dst[np.where(dst>0)] = ((h-l)*dst[np.where(dst>0)]-h*mini+l*maxi) / (maxi-mini)
        
    return dst


if __name__ == '__main__':
    # 注意：标注是顺时针方向，4个顶点
    
    # name = 'ydc'
    name = 'blw'
    root = './data/'+name
    for c in os.listdir(root):
        if '.json' in c:
            continue
        if '.npy' in c:
            continue
        
        img_path = os.path.join(root, c)
        anno_path = img_path.replace('.jpg','.json')
    
        img = plt.imread(img_path)
    
        f=open(anno_path,encoding='utf-8')
        anno = json.load(f)
        shapes = anno['shapes']
    
        region_map = generate_region_score(img,shapes)
        affinity_map = generate_affinity_score(img,shapes)
        np.save(os.path.join(root, name+'_region_'+(c.split('.')[0]).split('_')[1]+'.npy'), region_map)
        np.save(os.path.join(root, name+'_affinity_'+(c.split('.')[0]).split('_')[1]+'.npy'), affinity_map)
        print(c)