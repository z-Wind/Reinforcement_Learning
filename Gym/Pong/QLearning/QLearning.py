import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

from Gym.models.QLearningBase import QLearningBase


class QLearning(QLearningBase):
    def __init__(
        self,
        device,
        n_actions,
        n_features,
        learning_rate=0.01,
        gamma=0.9,
        tau=0.001,
        updateTargetFreq=10000,
        epsilonStart=1,
        epsilonEnd=0.2,
        epsilonDecayFreq=1000,
        mSize=10000,
        batchSize=200,
        startTrainSize=100,
        transforms=None,
    ):
        netEval = Net(n_features, n_actions)
        netTarget = Net(n_features, n_actions)

        # optimizer 是訓練的工具
        # 傳入 net 的所有參數, 學習率
        optimizer = torch.optim.Adam(netEval.parameters(), lr=learning_rate)

        super().__init__(
            device=device,
            netEval=netEval,
            netTarget=netTarget,
            optimizer=optimizer,
            n_actions=n_actions,
            learning_rate=learning_rate,
            gamma=gamma,
            tau=tau,
            updateTargetFreq=updateTargetFreq,
            epsilonStart=epsilonStart,
            epsilonEnd=epsilonEnd,
            epsilonDecayFreq=epsilonDecayFreq,
            mSize=mSize,
            batchSize=batchSize,
            startTrainSize=startTrainSize,
            transforms=transforms,
        )

    def choose_action(self, state):
        action = super().choose_action(state)

        return action, action


class Net(torch.nn.Module):
    def __init__(self, img_shape, n_actions):
        super(Net, self).__init__()
        # 定義每層用什麼樣的形式
        in_channels = img_shape[2]
        h = img_shape[0]
        w = img_shape[1]

        kernel_size = 8
        stride = 4
        padding = 0
        self.conv1 = torch.nn.Conv2d(
            in_channels, 32, kernel_size=kernel_size, stride=stride, padding=padding
        )
        h = (h + padding * 2 - kernel_size) // stride + 1
        w = (w + padding * 2 - kernel_size) // stride + 1

        # self.pool1 = torch.nn.MaxPool2d(2)  # 32 x (h-2)//2 x (w-2)//2
        # h //= 2
        # w //= 2

        kernel_size = 4
        stride = 2
        padding = 0
        self.conv2 = torch.nn.Conv2d(
            32, 64, kernel_size=kernel_size, stride=stride, padding=padding
        )
        h = (h + padding * 2 - kernel_size) // stride + 1
        w = (w + padding * 2 - kernel_size) // stride + 1

        kernel_size = 3
        stride = 1
        padding = 0
        self.conv3 = torch.nn.Conv2d(
            64, 64, kernel_size=kernel_size, stride=stride, padding=padding
        )
        h = (h + padding * 2 - kernel_size) // stride + 1
        w = (w + padding * 2 - kernel_size) // stride + 1

        # self.pool2 = torch.nn.MaxPool2d(2)  # 64 x ((h-2)//2-2)//2 x ((w-2)//2-2)//2
        # h //= 2
        # w //= 2

        self.fc1 = torch.nn.Linear(64 * h * w, 512)
        self.fc2 = torch.nn.Linear(512, n_actions)

        # self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x):  # 這同時也是 Module 中的 forward 功能
        # 正向傳播輸入值, 神經網絡分析出輸出值
        # x = self.pool1(F.relu(self.conv1(x)))
        # x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.shape[0], -1)
        # x = self.dropout(x)
        x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        x = self.fc2(x)

        return x
