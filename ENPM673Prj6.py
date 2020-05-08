import numpy as np
import torch
import pandas as pd

import os

print(torch.__version__)
import torch.nn as NN
import torch.nn.functional as F
import torchvision

print(torchvision.__version__)
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
import helper

from CNNBuild import *
from trainNN import *
from DataSetDev import CustomDatasetFromFile

from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import cv2

# print(cv2.__version__)
print('Import Initializations complete')

flag = False
prgRun = True


def main(prgRun):
    if __name__ == '__main__':
        # DCTrain = \
        #     CustomDatasetFromFile('dogs-vs-cats/train/')
        # DCTest = \
        #     CustomDatasetFromFile('dogs-vs-cats/test1/')

        seed = 1
        np.random.seed(seed)
        torch.manual_seed(seed)

        classes = ('0 cat', '1 dog')

        # train_loader = torch.utils.data.DataLoader(
        #     torchvision.datasets.CIFAR10('./cifardata', train=True, download=True,
        #                                  transform=transforms.Compose(
        #                                      [transforms.ToTensor(),
        #                                       transforms.Normalize((0.5,),
        #                                                            (0.5,))])),
        #     batch_size=128, shuffle=True)
        #
        # test_loader = torch.utils.data.DataLoader(torchvision.datasets.CIFAR10('./cifardata', train=False,
        #                                                                        transform=transforms.Compose(
        #                                                                            [transforms.ToTensor(),
        #                                                                             transforms.Normalize((0.5,),
        #                                                                                                  (0.5,))])),
        #                                           batch_size=64, shuffle=True)

        # train_loader = torch.utils.data.DataLoader(dataset=DCTrain,
        #                                            batch_size=64,
        #                                            shuffle=True)
        #
        # test_loader = torch.utils.data.DataLoader(dataset=DCTest,
        #                                           batch_size=64,
        #                                           shuffle=True)
        #
        data_dir = './dogs-vs-cats'

        train_dir = "./dogs-vs-cats/train"
        train_dogs_dir = f'{train_dir}/dogs'
        train_cats_dir = f'{train_dir}/cats'
        #############################################

        # Find a way to create the subdirectories here.

        #############################################
        imagesize = 5

        train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                               transforms.Resize((imagesize, imagesize), interpolation=2),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])])

        test_transforms = transforms.Compose([transforms.Resize((imagesize, imagesize), interpolation=2),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])])

        # Pass transforms in here, then run the next cell to see how the transforms look
        test_data = datasets.ImageFolder(data_dir + '/test1/', transform=test_transforms)
        train_data = datasets.ImageFolder(data_dir + '/train/', transform=train_transforms)

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=len(train_data), shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=len(test_data), shuffle=True)

        # data_iter = iter(train_loader)
        #
        # images, labels = next(data_iter)
        # fig, axes = plt.subplots(figsize=(10, 10), ncols=4)
        #
        # image=images[0].permute(1, 2, 0).numpy()
        #
        # cv2.imshow('image',image)

        CNN = CNNBuild()
        trainNN(CNN, batch_size=1028, epochs=2, lr=.1, train_loader=train_loader, test_loader=test_loader,
                classes=classes)

        prgRun = False
        return prgRun

        prgRun = False
        return prgRun


print('Function Initializations complete')

if __name__ == '__main__':
    print('Start Program')
    prgRun = True
    while prgRun == True:
        prgRun = main(prgRun)

    print('Goodbye!')
    cv2.destroyAllWindows()
