import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

from Gym.models.PolicyGradientBase import PolicyGradientBase


class PolicyGradient(PolicyGradientBase):
    def __init__(
        self,
        device,
        n_actions,
        n_features,
        learning_rate=0.01,
        gamma=0.9,
        mSize=10000,
        transforms=None,
    ):
        net = Net(n_features, n_actions)

        # optimizer 是訓練的工具
        # 傳入 net 的所有參數, 學習率
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

        super().__init__(
            device=device,
            net=net,
            optimizer=optimizer,
            n_actions=n_actions,
            learning_rate=learning_rate,
            gamma=gamma,
            mSize=mSize,
            transforms=transforms,
        )

    def choose_action(self, state):
        action, _ = super().choose_action(state)

        m = Categorical(action)
        a = m.sample()
        log_prob = m.log_prob(a)

        action = action.cpu().data.numpy()

        return (action, log_prob), a.item()


class Net(torch.nn.Module):
    def __init__(self, n_features, n_actions):
        super(Net, self).__init__()
        # 定義每層用什麼樣的形式
        self.fc1 = torch.nn.Linear(n_features, 128)
        self.fc2 = torch.nn.Linear(128, n_actions)  # Prob of Left

    def forward(self, x):  # 這同時也是 Module 中的 forward 功能
        # 正向傳播輸入值, 神經網絡分析出輸出值
        model = torch.nn.Sequential(
            self.fc1, torch.nn.ReLU(), self.fc2, torch.nn.Softmax(dim=1)
        )
        return model(x)
