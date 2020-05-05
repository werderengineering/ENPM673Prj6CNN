from __main__ import *


class CNNBuild(NN.Module):

    def __init__(self):
        super(CNNBuild, self).__init__()

        self.conv1 = NN.Conv2d(3, 18, 3, 1, 1)
        self.conv2 = NN.Conv2d(18, 128, 5, 1)
        self.pool = NN.MaxPool2d(2, 2, 0)
        self.fc1 = NN.Linear(4608, 2304)
        self.fc2 = NN.Linear(2304, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        # x = self.conv2(x)
        # x = F.relu(x)
        # x = self.pool(x)

        x = x.view(-1, 4608)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        # x = F.log_softmax(x, dim=1)

        return (x)

