#-*- coding:utf-8 -*-
"""
    desc: 将标注好的数据集里面的四种文件分别放入对应的子目录，如下所示：
            affinity：blw_affinity_1.npy
            anno：blw_1.json
            img：blw_1.jpg
            region：blw_region_1.npy

    author:MeteorMan
    datetime:2020/11/16
"""

import os
from shutil import copy

paths = ['affinity', 'anno', 'img', 'region']
path = './data'

for p in paths:
    sub_p = os.path.join(path, p)
    if not os.path.exists(sub_p):
        os.mkdir(sub_p)

data_path = './data/blw'
for file in os.listdir(data_path):
    # print(file)
    if file.endswith('.jpg'):
        copy(os.path.join(data_path, file), os.path.join(path, 'img'))
    if file.endswith('json'):
        copy(os.path.join(data_path, file), os.path.join(path, 'anno'))
    if file.endswith('npy') and 'affinity' in file:
        copy(os.path.join(data_path, file), os.path.join(path, 'affinity'))
    if file.endswith('npy') and 'region' in file:
        copy(os.path.join(data_path, file), os.path.join(path, 'region'))
