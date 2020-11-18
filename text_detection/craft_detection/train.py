from torch.utils.data import DataLoader
from my_dataset import MyDataset
from my_model import CRAFT
import torch.nn as nn
import torch
import os
from collections import OrderedDict
import numpy as np


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

if __name__ == '__main__':
    """参数设置"""
    # device = 'cuda' # cpu 或 cuda
    device = 'cpu' # cpu 或 cuda
    dataset_path = './data' # 自己数据集的路径
    pretrained_path = './pretrained/craft_mlt_25k.pth' # 预训练模型的存放路径
    model_path = './models' # 现在训练的模型要存储的路径
    
    
    dataset = MyDataset(dataset_path)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    net = CRAFT(phase='train').to(device)
    net.load_state_dict(copyStateDict(torch.load(pretrained_path, map_location=device)))
    criterion=nn.MSELoss(size_average=False).to(device)
    optimizer=torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),1e-7,
                              momentum=0.95,
                              weight_decay=0)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
        
    for epoch in range(500):
        epoch_loss = 0
        for i, data in enumerate(loader):
            img = data['img'].to(device)
            gt = data['gt'].to(device)
            
            # forward
            y, _ = net(img)
            loss = criterion(y, gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach()
        print('epoch loss_'+str(epoch),':',epoch_loss/len(loader))
        torch.save(net.state_dict(), os.path.join(model_path,str(epoch)+'.pth'))
            