import numpy as np
import torch
import pandas as pd
import re
import shutil

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
from FileDev import *

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
        test_dir = "./dogs-vs-cats/test1/test1"
        subdir = '/train'
        SubDirectories(train_dir, subdir)

        # More information the higher the number, max around 230
        imagesize = 30

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

        trainBatch = int(len(test_data) * .3)
        trainBatch = 64

        testBatch = int(len(test_data))

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=trainBatch, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=testBatch, shuffle=False)

        CNN = CNNBuild()
        loadModel = str.lower(input('\nLoad a model? Enter |yes| or |no|: '))
        if loadModel == 'yes' or loadModel == 'y':
            name = str.lower(input('Enter the entire name: '))
            if name == '':
                name = 'model12810x10.1lr.pth'
            CNN.load_state_dict(torch.load(name))
            CNN.eval()
            print('Model loaded: ', name)

        else:
            trainNN(CNN, batch_size=trainBatch, epochs=128, lr=.1, train_loader=train_loader, test_loader=test_loader,
                    classes=classes)

            saveModelYN = str.lower(input('\nSave the model? Enter |yes| or |no|: '))
            if saveModelYN == 'yes' or saveModelYN == 'y':
                name = str.lower(input('Enter a name: '))
                torch.save(CNN.state_dict(), 'model' + name + '.pth')

        print('\nTesting Model')
        testCNN(CNN, batch_size=testBatch, epochs=1, lr=.1, train_loader=train_loader, test_loader=test_loader,
                classes=classes, directory=test_dir)

        getdataYN = str.lower(input('\nRestore folders? Enter |yes| or |no|: '))
        if getdataYN == 'yes' or getdataYN == 'y':
            print('Please wait for the program to restore all the files')
            restoreSubdirectories(train_dir, subdir)

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
