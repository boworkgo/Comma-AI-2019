import torch
import torch.nn as nn
import torch.nn.functional as F
from train import Net
import cv2 as cv
import numpy as np
from torch.autograd import Variable
from sklearn.metrics import mean_squared_error
import pandas as pd

model = Net()
model.load_state_dict(torch.load("./trained.pth"))


def new(no_img, name, start=1):
    images = np.array(
        [
            cv.imread("../data/frames/{}_farne_{}.jpg".format(name, i))
            for i in range(start, no_img)
        ]
    )
    images = images.reshape(-1, 3, 128, 128)
    images = torch.from_numpy(images).to(torch.float)
    images = Variable(images)

    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    return _


def write_test():
    for i in range(14):
        e = 816 * (i + 1) if i != 13 else 816 * i + 189
        predicted = new(e, "test", 816 * i)
        with open("../data/test.txt", "a") as f:
            for j, p in enumerate(predicted):
                p = p.data.numpy()
                f.write(str(p) + "\n")
                if i == 0 and j == 0:
                    f.write(str(p) + "\n")


def calculate_mse(no_frames):
    predicted = new(no_frames, "train")
    ans = np.array(
        pd.read_csv("../data/train.txt", sep="\n", header=None)[0].values[1:no_frames]
    )
    predicted = predicted.detach().numpy()
    mse = mean_squared_error(ans, predicted)
    print("MSE: {}".format(mse))


# write_test()
calculate_mse(6000)  # 6.7
