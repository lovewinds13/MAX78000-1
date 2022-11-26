import os
import glob 
# glob是python自己带的一个文件操作相关模块，用它可以查找符合自己目的的文件，类似于Windows下的文件搜索，支持通配符操作

import random
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

# 改写所需文件
import ai8x
import numpy as np

class ImageFolder(Dataset):
    def __init__(self, image_dir, image_size, transform=True):
        # image_size将图像裁剪为指定大小
        self.image_size = image_size
        self.image_paths = []
        self.image_labels = []
        self.transform = transform

        # classes通过listdir得到train/test文件夹下的各个文件夹，用于取出图像和作为标签
        # 打印classes:['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
        self.classes = sorted(os.listdir(image_dir))

        # 通过以下得到图像和标签
        for idx, cls_ in enumerate(self.classes):
            # image_paths每一张图像的路径。
            #如：image_paths:['/home/lyl/MAX78000/ai8x-training/data/garbage/train/cardboard/cardboard236.jpg'
            self.image_paths += glob.glob(os.path.join(image_dir, cls_, '*.*'))
            # image_labels标签
            # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,..........]
            self.image_labels += [idx] * len(glob.glob(os.path.join(image_dir, cls_, '*.*')))
        # 有多少图像
        self.indexes = list(range(len(self.image_paths)))

    '''__getitem__'''
    def __getitem__(self, index):
        # 获取具体的每张图像和相应的标签
        """
        2
        /home/lyl/MAX78000/ai8x-training/data/garbage/train/metal/metal228.jpg
        """
        image_path = self.image_paths[self.indexes[index]]
        image_label = self.image_labels[self.indexes[index]]
        image = Image.open(image_path).resize(self.image_size)
        # 对图像作预处理
        random.shuffle(self.indexes)
        if self.transform is not None:
            image = self.transform(image)

        return image, image_label

    '''__len__'''
    def __len__(self):
        return len(self.indexes)


def garbage_get_datasets(data, load_train=True, load_test=True):
    # 数据文件夹位置
    (data_dir, args) = data
    
    # 要定制的图像大小
    image_size = (224, 224)
    
    # 对训练数据进行预处理
    if load_train:
        data_path =data_dir+'/garbage/train'
        
#         random.shuffle(self.indexes)
        
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=1, contrast=1, saturation=0.5, hue=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        train_dataset = ImageFolder(data_path, image_size, train_transform)
        
    # 对测试数据进行预处理
    if load_test:
        data_path =data_dir+'/garbage/test'
        
#         random.shuffle(self.indexes)
        
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        test_dataset = ImageFolder(data_path, image_size, train_transform)
        
        
        if args.truncate_testset:
            test_dataset.data = test_dataset.data[:1]
    else:
        test_dataset = None

    return train_dataset, test_dataset


datasets = [
    {
        'name': 'garbage',
        'input': (3, 224, 224),
        'output': ('cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash'),
        'loader': garbage_get_datasets,
    },
]

