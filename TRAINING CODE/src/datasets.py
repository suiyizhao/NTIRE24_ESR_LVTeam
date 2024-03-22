import os
import glob
import torch
import random

import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset, DataLoader

class PairedImgDataset(Dataset):
    def __init__(self, data_source, mode, crop=256):
        assert mode in ['train', 'val', 'test']

        self.data_source = data_source
        self.crop = crop
        self.mode = mode
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        if self.mode == 'train':
            self.img_paths = sorted(glob.glob(self.data_source + '/DIV2K/DIV2K_train_LR_bicubic' + '/*.*')) * 85 + sorted(glob.glob(self.data_source + '/LSDIR/train/LRx4/*/*.*'))
            self.gt_paths = sorted(glob.glob(self.data_source + '/DIV2K/DIV2K_train_HR' + '/*.*')) * 85 + sorted(glob.glob(self.data_source + '/LSDIR/train/HR/*/*.*'))
        if self.mode == 'val' or self.mode == 'test':
            self.img_paths = sorted(glob.glob(self.data_source + '/LSDIR_DIV2K/LSDIR_DIV2K_valid_LR' + '/*.*'))
            self.gt_paths = sorted(glob.glob(self.data_source +  '/LSDIR_DIV2K/LSDIR_DIV2K_valid_HR' + '/*.*'))
            
    def __getitem__(self, index):
        
        name = os.path.basename(self.img_paths[index % len(self.img_paths)])
        
        img = Image.open(self.img_paths[index % len(self.img_paths)]).convert('RGB')
        gt = Image.open(self.gt_paths[index % len(self.gt_paths)]).convert('RGB')
        
        if self.mode == 'train':
            
            # crop
            width, height = img.size
            offset_w = random.randint(0, max(0, width - self.crop - 1))
            offset_h = random.randint(0, max(0, height - self.crop - 1))
            
            img = img.crop((offset_w, offset_h, offset_w + self.crop, offset_h + self.crop))
            gt = gt.crop((offset_w*4, offset_h*4, offset_w*4 + self.crop*4, offset_h*4 + self.crop*4))
            
            # horizontal flip
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                gt = gt.transpose(Image.FLIP_LEFT_RIGHT)
            
        img = self.transform(img)
        gt = self.transform(gt)

        if self.mode == 'train' or self.mode == 'val':
            return img, gt
        else:
            return img, gt, name

    def __len__(self):
        return max(len(self.img_paths), len(self.gt_paths))

class SingleImgDataset(Dataset):
    def __init__(self, data_source):
        
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        self.img_paths = sorted(glob.glob(data_source + '/' + 'test' + '/blurry' + '/*/*.*'))

    def __getitem__(self, index):                    
                              
        path = self.img_paths[index % len(self.img_paths)]
        
        name = os.path.basename(path)
        
        img = Image.open(path).convert('RGB')
        
        img = self.transform(img)
        
        return img, name

    def __len__(self):
        return len(self.img_paths)
