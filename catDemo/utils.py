# --coding:utf-8--
import torch
import os
from config import Dataset
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
# ------------------------------------------------ #
# 1.数据集处理
# ------------------------------------------------ #
def save_file(path, data):
    with open(path, "w", encoding="utf-8") as f:
        f.write(data)


def load_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class MyDataset(torch.utils.data.Dataset):  # 创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
    def __init__(self, root, text_path,  transform=None):  # 初始化一些需要传入的参数
        '''
        :param root: 数据集目录
        :param classes_num: the num of category
        :param transform:
        '''
        # info = []
        self.root = root
        # self.text_path = text_path
        self.transform = transform
        self.info_list = self.file2list(text_path)

    def __getitem__(self, index):# 这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        return self.pull_item(index)

    def __len__(self):  # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.info_list)

    def file2list(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read().split('\n')

    def pull_item(self, index):
        # print(self.info_list[index])
        path, label = self.info_list[index].split('\t')
        label = int(label)
        path = self.root + '/' + path
        # print(path)
        img = Image.open(path)
        img = img.convert("RGB")
        # print(img)
        # img.show()
        # img = np.array(img)
        if self.transform != None:
            img = self.transform(img)
        return img/255.0, label


def get_dataset(root, trainTxtPath, trainPercent, batchsize, transform=None):
    data = MyDataset(root, trainTxtPath,  transform=transform)
    train_size = int(len(data) * trainPercent)
    test_size = len(data) - train_size
    print(train_size, test_size)
    train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size])
    train_data = DataLoader(
        train_dataset,
        batch_size=batchsize,
        shuffle=True
    )
    test_data = DataLoader(
        test_dataset,
        batch_size=batchsize,
        shuffle=True
        # shuffle=shuffle
    )
    return train_data, test_data

# dataset = Dataset()
# dataPath = dataset.ImgPath
# trainTxtPath = dataset.TrainTxtPath
# trainPercent = 0.8
# batchsize = 32
# D_TRAIN, D_TEST = get_dataset(dataPath, trainTxtPath, trainPercent, batchsize)



