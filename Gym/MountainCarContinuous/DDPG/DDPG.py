import torch
import torch.nn.functional as F
import numpy as np

from Gym.models.DDPGBase import DDPGBase


class DDPG(DDPGBase):
    def __init__(
        self,
        device,
        n_actions,
        max_action,
        n_features,
        noiseStart=2,
        noiseEnd=0.2,
        learning_rate=0.01,
        gamma=0.9,
        tau=0.001,
        noiseDecayFreq=10 ** 5,
        updateTargetFreq=10000,
        mSize=10000,
        batchSize=200,
        startTrainSize=100,
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
            noiseStart=noiseStart,
            noiseEnd=noiseEnd,
            noiseDecayFreq=noiseDecayFreq,
            updateTargetFreq=updateTargetFreq,
            mSize=mSize,
            batchSize=batchSize,
            startTrainSize=startTrainSize,
            transforms=transforms,
        )

        self.max_action = max_action

    def choose_action(self, state):
        action = super().choose_action(state)
        action = np.clip(action, -1, 1)

        a = action[0] * self.max_action
        return action, a


class ActorNet(torch.nn.Module):
    def __init__(self, n_actions, n_features):
        super(ActorNet, self).__init__()
        # 定義每層用什麼樣的形式
        self.fc1 = torch.nn.Linear(n_features, 128)
        self.fc2 = torch.nn.Linear(128, n_actions)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = torch.tanh(self.fc2(x))

        return x


class CriticNet(torch.nn.Module):
    def __init__(self, n_actions, n_features):
        super(CriticNet, self).__init__()
        # 定義每層用什麼樣的形式
        self.fcVal1_s = torch.nn.Linear(n_features, 256)
        self.fcVal2_s = torch.nn.Linear(256, 128)

        self.fcVal1_a = torch.nn.Linear(n_actions, 128)

        self.fcVal3 = torch.nn.Linear(256, 128)
        self.fcVal4 = torch.nn.Linear(128, 1)

    def forward(self, x, a):  # 這同時也是 Module 中的 forward 功能
        # 正向傳播輸入值, 神經網絡分析出輸出值
        x_v = F.relu(self.fcVal1_s(x))
        x_v = F.relu(self.fcVal2_s(x_v))

        x_a = F.relu(self.fcVal1_a(a))

        x = torch.cat((x_v, x_a), dim=1)
        x = F.relu(self.fcVal3(x))
        qVal = self.fcVal4(x)

        return qVal


class ActorCriticNet(torch.nn.Module):
    def __init__(self, n_actions, n_features):
        super(ActorCriticNet, self).__init__()
        # 定義每層用什麼樣的形式
        self.actor = ActorNet(n_actions, n_features)
        self.critic = CriticNet(n_actions, n_features)

    def forward(self, x):  # 這同時也是 Module 中的 forward 功能
        # 正向傳播輸入值, 神經網絡分析出輸出值
        action = self.actor(x)
        qVal = self.critic(x, action)

        return action, qVal

    def action(self, x):
        action = self.actor(x)

        return action

    def criticism(self, x, a):
        qVal = self.critic(x, a)

        return qVal
