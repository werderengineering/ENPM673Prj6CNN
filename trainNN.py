import time
import datetime
import torch.optim as optim
from torch.autograd import Variable
from precision import *
import numpy as np
import matplotlib.pyplot as plt
import winsound
from imageFileNames import *
import cv2
from imutils import build_montages

font = cv2.FONT_HERSHEY_SIMPLEX
import random

from __main__ import *


# def trainNet(net, batch_size, epochs, lr,train_set,train_sampler,val_sampler,test_set,test_sampler, classes):
def trainNN(net, batch_size, epochs, lr, train_loader, test_loader, classes):
    print("\nHyper Parameters")
    print("############################################")
    print("batch size: ", batch_size)
    print("epochs: ", epochs)
    print("learning rate: ", lr)
    print("############################################")
    # winsound.Beep(1000, 500)

    n_batches = len(train_loader)

    # loss = F.nll_loss
    loss = NN.CrossEntropyLoss()

    # optimizer = optim.Adam(net.parameters(), lr)
    optimizer = optim.Adadelta(net.parameters(), lr)

    StartTrainTime = time.time()

    valcount = 0

    trials = 0
    trialVset = []
    lossVset = []
    accuracyVset = []

    trialTset = []
    lossTset = []
    accuracyTset = []
    epochSet = []

    for epoch in range(epochs):

        indiTrain = np.zeros(len(classes))
        TotTrain = np.zeros(len(classes))

        runloss = 0.0
        print_every = n_batches // 10
        start_time = time.time()
        totalLoss = 0

        print("Validating...")
        ValidationTotalLoss = 0
        for inputs, labels in train_loader:
            inputs, labels = Variable(inputs), Variable(labels)

            # print(inputs.shape)

            ValO = net(inputs)

            LossVal = loss(ValO, labels)
            ValidationTotalLoss += LossVal.data
            ValLossOut = float(ValidationTotalLoss / len(train_loader))

        trialVset.append(trials)
        lossVset.append(ValLossOut)
        print("Validation loss: ", ValLossOut)
        # winsound.Beep(2500, 250)

        # if valcount>1:
        #     if ValLossOutP<ValLossOut:
        #         print("Avoiding over learning")
        #         break
        # valcount+=1
        # ValLossOutP=ValLossOut

        # if ValLossOut<.9:
        #     print("Val loss less than 1")
        #     break

        for i, data in enumerate(train_loader, 0):
            trials += 1

            inputs, labels = data

            inputs, labels = Variable(inputs), Variable(labels)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss_size = loss(outputs, labels)
            loss_size.backward()
            optimizer.step()

            runloss += loss_size.data
            totalLoss += loss_size.data

            if i % 100 == 0:
                trialTset.append(trials)
                lossTset.append(totalLoss)

                print("\nEpoch:", epoch + 1)
                print("Loss: ", float(runloss / print_every))
                print("Time: ", str(datetime.timedelta(seconds=time.time() - start_time)))
                running_loss = 0.0
                start_time = time.time()

        _, predicted = torch.max(outputs.data, 1)
        predicted = np.asarray(predicted).astype(int)
        labels = np.asarray(labels).astype(int)

        for i in range(len(predicted)):
            predictedi = predicted[i]
            labelsi = labels[i]

            indiTrain, TotTrain, PR = precisionget(predictedi, labelsi, indiTrain, TotTrain)
        print("Epoch: ", epoch + 1)
        print("Total trials on each: ", TotTrain)
        print("Individual Reporting:", PR)
        print("Average Correct Percent:", np.sum(PR) / len(TotTrain))
        accuracyT = np.sum(PR) / len(TotTrain)
        accuracyTset.append(accuracyT)
        epochSet.append(epoch + 1)

    print("Time to Train: ", str(datetime.timedelta(seconds=time.time() - start_time)))
    print("Training Confusion Table")
    print(predicted)
    print(labels)

    print("Total Training Time: ", str(datetime.timedelta(seconds=time.time() - StartTrainTime)))
    winsound.Beep(2500, 1000)

    plt.plot(epochSet, accuracyTset)
    plt.title('Average Accuracy per Epoch')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.show()

    plt.plot(trialTset, lossTset)
    plt.title('Training Cross entropy loss')
    plt.ylabel('Error')
    plt.xlabel('Trials')
    plt.show()

    plt.plot(trialVset, lossVset)
    plt.title('Validation Cross entropy loss')
    plt.ylabel('Error')
    plt.xlabel('Trials')
    plt.show()


def testCNN(net, batch_size, epochs, lr, train_loader, test_loader, classes, directory):
    # test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, sampler=test_sampler, num_workers=2)
    indiTest = np.zeros(len(classes))
    TotTest = np.zeros(len(classes))

    for inputs, labels in test_loader:
        inputs, labels = Variable(inputs), Variable(labels)

        TestO = net(inputs)
        _, predicted = torch.max(TestO.data, 1)

        predicted = np.array(predicted).astype(int)
        labels = np.array(labels).astype(int)

        for i in range(len(predicted)):
            predictedi = predicted[i]
            labelsi = labels[i]

            # print("Predicted Labels: ", classes[predictedi])
            # print("Test Label: ", classes[labelsi])

            indiTest, TotTest, PR = precisionget(predictedi, labelsi, indiTest, TotTest)

    print('\nTest Accuracy Results:\n')
    print("Total trials on each: ", TotTest)
    print("Individual Reporting:", PR)
    print("Average Correct Percent:", np.sum(PR) / len(TotTest))
    print("Test Confusion Table")
    _, top = torch.topk(TestO, 2)
    # print(top)
    print(predicted)
    print(labels)

    imageList = imagefiles(directory)
    images = []

    randval = random.randint(0, len(predicted) - 26)

    for i in range(randval, randval + 26):

        frameDir = directory + '/' + imageList[i]
        frame = cv2.imread(frameDir)
        frame = cv2.resize(frame, (128, 128))
        labelDC = int(predicted[i])
        if labelDC == 0:
            labelOut = 'Cat'
            cv2.putText(frame, labelOut, (5, 15), font, 0.5, (0, 0, 255), 2)
        else:
            labelOut = 'Dog'
            cv2.putText(frame, labelOut, (5, 15), font, 0.5, (0, 255, 0), 2)
        images.append(frame)
        montage = build_montages(images, (128, 128), (5, 5))[0]

        cv2.imshow('Animal', montage)
        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break
