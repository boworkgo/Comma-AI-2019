import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
import cv2 as cv
import numpy as np
from sklearn.model_selection import train_test_split
import random
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
from math import ceil
import time


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(48 * 16 * 16, 1164)
        self.fc2 = nn.Linear(1164, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear(50, 10)
        self.fc5 = nn.Linear(10, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        # print(x.size())
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        return x


def train(model):
    train_x_all = np.array(
        [
            cv.imread("../data/frames/train_farne_{}.jpg".format(i))
            for i in range(1, 20399)
        ]
    )
    train_y_all = np.array(
        pd.read_csv("../data/train.txt", sep="\n", header=None)[0].values[1:20399]
    )
    batch_size = 64
    optimzer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()

    for epoch in range(6): # 25
        train_x_all, train_y_all = shuffle(train_x_all, train_y_all)
        print("[*] Loaded dataset")
        one_val_x, one_val_y = None, None
        start_time = time.time()
        no_batches = ceil(20398 / batch_size)

        for batch in range(no_batches):
            b, e = batch * batch_size, (batch + 1) * batch_size
            if e > 20398:
                e = 20398
            train_x, train_y = train_x_all[b:e], train_y_all[b:e]

            train_x, val_x, train_y_ult, val_y_ult = train_test_split(
                train_x, train_y, test_size=0.2
            )

            train_x = train_x.reshape(-1, 3, 128, 128)
            train_x = torch.from_numpy(train_x).to(torch.float)
            train_y = torch.from_numpy(train_y_ult).to(torch.float)
            train_y = train_y.view(-1, 1)

            val_x = val_x.reshape(-1, 3, 128, 128)
            val_x = torch.from_numpy(val_x).to(torch.float)
            val_y = torch.from_numpy(val_y_ult).to(torch.float)
            val_y = val_y.view(-1, 1)

            model.train()
            x_train, y_train = Variable(train_x), Variable(train_y)
            x_val, y_val = Variable(val_x), Variable(val_y)
            if batch == 0:
                one_val_x, one_val_y = val_x, val_y
            else:
                oneval_x = torch.cat((one_val_x, val_x), 0)
                oneval_y = torch.cat((one_val_y, val_y), 0)

            optimzer.zero_grad()
            output_train = model(x_train)
            loss = criterion(output_train, y_train)
            loss.backward()
            optimzer.step()
            if (batch + 1) % (no_batches // 3) == 0:
                print(
                    "\t[>] \t{} secs\tTrain loss of {}".format(
                        time.time() - start_time, loss
                    )
                )

        output_val = model(one_val_x)
        loss = criterion(output_val, one_val_y)
        print("\t\t[%] Epoch {}\tLoss val of {}".format(epoch + 1, loss))
    return model


def module():
    model = train(Net())
    torch.save(model.state_dict(), "./trained.pth")


# module()
