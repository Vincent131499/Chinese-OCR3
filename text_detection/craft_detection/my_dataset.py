from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt
import matplotlib.cm as CM
import numpy as np
import torch
import cv2


class MyDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.imglist = [f.split('.')[0] for f in os.listdir(os.path.join(root, 'img'))]
    
    def __getitem__(self, index):
        # read img, region_map, affinity_map
        img_path = os.path.join(self.root, 'img', self.imglist[index]+'.jpg')
#        img = plt.imread(img_path)
        img = np.array(plt.imread(img_path))
        
        region_path = os.path.join(self.root, 'region', 
                                   self.imglist[index].split('_')[0]+'_region_'
                                   +self.imglist[index].split('_')[1]+'.npy')
        region_map = np.load(region_path).astype(np.float32)
        
        affinity_path = os.path.join(self.root, 'affinity', 
                                   self.imglist[index].split('_')[0]+'_affinity_'
                                   +self.imglist[index].split('_')[1]+'.npy')
        affinity_map = np.load(affinity_path).astype(np.float32)
        
        # 保证图像长和宽是2的倍数
        h, w, c = img.shape
        if h % 2 != 0 or w % 2 != 0:
            h = int(h // 2 * 2)
            w = int(w // 2 * 2)
            img = cv2.resize(img, (w, h))
            region_map = cv2.resize(region_map, (w, h))
            affinity_map = cv2.resize(affinity_map, (w, h))
        
        # preprocess
        img = normalizeMeanVariance(img)
        img = torch.from_numpy(img).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
        
        region_map = cv2.resize(region_map, (w//2, h//2))
        region_map = torch.tensor(region_map).unsqueeze(2)
        affinity_map = cv2.resize(affinity_map, (w//2, h//2))
        affinity_map = torch.tensor(affinity_map).unsqueeze(2)
        gt_map = torch.cat((region_map,affinity_map), dim=2)
        
        return {'img':img, 'gt':gt_map}
        
    
    def __len__(self):
        return len(self.imglist)

def normalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    # should be RGB order
    img = in_img.copy().astype(np.float32)
    img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
    img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)
    return img
  

if __name__ == '__main__':
    d = MyDataset('./blw')
    for i, data in enumerate(d):
        img = data['img']
#        plt.imshow(img)
#        plt.figure()
        gt = data['gt']
#        plt.imshow(region,cmap=CM.jet)
        print(gt.max())
        print(gt.shape)
        break
    