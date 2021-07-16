from torch.utils.data import Dataset
from PIL import Image
import torch

import torchvision.transforms as transforms
import numpy as np
import config
import time


def readTxt(file_path):  
    img_list = []
    with open(file_path, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
            item = lines.strip().split()  
            img_list.append(item)
    file_to_read.close()
    return img_list
#single frame
class GetData(Dataset):
    def __init__(self, file_path, transforms):
        self.img_list = readTxt(file_path)
        self.dataset_size = len(self.img_list)
        self.transforms = transforms

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        img_path_list = self.img_list[idx]
        data = Image.open(img_path_list[0])
        label = Image.open(img_path_list[1])
        data = self.transforms(data)

        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        label = torch.squeeze(transform(label))
        # label = torch.Tensor(label)
        sample = {'data': data, 'label': label}
        return sample
class RoadDataset(Dataset): 
    def __init__(self, file_path, transforms):
        self.img_list = readTxt(file_path)
        self.dataset_size = len(self.img_list)
        self.transforms = transforms

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        img_path_list = self.img_list[idx]
        data = Image.open(img_path_list[4])
        label = Image.open(img_path_list[5])
        data = self.transforms(data)
        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        label = torch.squeeze(transform(label))
        sample = {'data': data, 'label': label}
        return sample
#multi frames

class RoadDatasetList(Dataset):

    def __init__(self, file_path, transforms,x,y):

        self.img_list = readTxt(file_path)
        self.dataset_size = len(self.img_list)
        self.transforms = transforms
        self.x = x
        self.y = y
    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        img_path_list = self.img_list[idx]
        data = []
        if len(img_path_list) < self.x:
            return
        for i in range(self.x):
            data.append(torch.unsqueeze(self.transforms(Image.open(img_path_list[i])), dim=0))
        data = torch.cat(data, 0) 
        
        label = Image.open(img_path_list[self.y])
        label = torch.squeeze(self.transforms(label))
        sample = {'data': data, 'label': label}
        return sample
