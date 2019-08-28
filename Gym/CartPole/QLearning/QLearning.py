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
    def __init__(self, n_features, n_actions):
        super(Net, self).__init__()
        # 定義每層用什麼樣的形式
        self.fc1 = torch.nn.Linear(n_features, 10)
        self.fc2 = torch.nn.Linear(10, n_actions)  # Prob of Left

    def forward(self, x):  # 這同時也是 Module 中的 forward 功能
        # 正向傳播輸入值, 神經網絡分析出輸出值
        model = torch.nn.Sequential(self.fc1, torch.nn.ReLU6(), self.fc2)

        return model(x)
