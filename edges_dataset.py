import os
import torch
from torchvision import transforms
from PIL import Image
from PIL import ImageOps
import numpy as np


mypath_A = 'C:/dataset/edges2handbags/train/'
mypath_B = 'C:/dataset/edges2handbags/val/'



class imgldr(torch.utils.data.Dataset):
    def __init__(self,batch_size,path=mypath_A,resize_img = 256):
         self.path = path
         self.bn   = batch_size
         self.cnt  = 0
         self.getlist(path)
         self.resize_img = resize_img
         self.transform = transforms.Compose([transforms.Scale(resize_img), transforms.ToTensor() ])


    def getlist(self,path):
         self.list = [f for f in os.listdir(self.path)]
         self.lens = len(self.list)

    def __len__(self):
        return self.lens

    def __getitem__(self, index):
        image = Image.open(self.path + self.list[index]).convert('RGB')
        return self.transform(image)
