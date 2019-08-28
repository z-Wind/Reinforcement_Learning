import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

from Gym.models.A2CBase import A2CBase


class A2C(A2CBase):
    def __init__(
        self,
        device,
        n_actions,
        n_features,
        learning_rate=0.01,
        gamma=0.9,
        tau=0.001,
        updateTargetFreq=10000,
        mSize=10000,
        batchSize=200,
        transforms=None,
    ):
        actorCriticEval = ActorCriticNet(n_actions, n_features)
        actorCriticTarget = ActorCriticNet(n_actions, n_features)

        # optimizer 是訓練的工具
        # 傳入 net 的所有參數, 學習率
        optimizerCritic = torch.optim.Adam(
            actorCriticEval.critic.parameters(), lr=learning_rate
        )
        optimizerActor = torch.optim.Adam(
            actorCriticEval.actor.parameters(), lr=learning_rate
        )

        super().__init__(
            device=device,
            actorCriticEval=actorCriticEval,
            actorCriticTarget=actorCriticTarget,
            optimizerCritic=optimizerCritic,
            optimizerActor=optimizerActor,
            n_actions=n_actions,
            learning_rate=learning_rate,
            gamma=gamma,
            tau=tau,
            updateTargetFreq=updateTargetFreq,
            mSize=mSize,
            batchSize=batchSize,
            transforms=transforms,
        )

    def choose_action(self, state):
        action, _ = super().choose_action(state)

        m = Categorical(action)
        a = m.sample()

        log_prob = m.log_prob(a)

        action = action.cpu().data.numpy()

        return (action, log_prob), a.item()


class ActorNet(torch.nn.Module):
    def __init__(self, n_actions, n_features):
        super(ActorNet, self).__init__()
        # 定義每層用什麼樣的形式
        self.fc1 = torch.nn.Linear(n_features, 128)
        self.fc2 = torch.nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)

        return x


class CriticNet(torch.nn.Module):
    def __init__(self, n_features):
        super(CriticNet, self).__init__()
        # 定義每層用什麼樣的形式
        self.fc1 = torch.nn.Linear(n_features, 256)
        self.fc2 = torch.nn.Linear(256, 1)

    def forward(self, x):  # 這同時也是 Module 中的 forward 功能
        # 正向傳播輸入值, 神經網絡分析出輸出值
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class ActorCriticNet(torch.nn.Module):
    def __init__(self, n_actions, n_features):
        super(ActorCriticNet, self).__init__()
        # 定義每層用什麼樣的形式
        self.actor = ActorNet(n_actions, n_features)
        self.critic = CriticNet(n_features)

    def forward(self, x):  # 這同時也是 Module 中的 forward 功能
        # 正向傳播輸入值, 神經網絡分析出輸出值
        action = self.actor(x)
        val = self.critic(x)

        return action, val

    def action(self, x):
        action = self.actor(x)

        return action

    def criticism(self, x):
        val = self.critic(x)

        return val
