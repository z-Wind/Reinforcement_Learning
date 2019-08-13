import torch
import torch.nn.functional as F
from torch.distributions import Normal
from .memory import MemoryDataset
from collections import namedtuple
import numpy as np

torch.manual_seed(500)  # 固定隨機種子 for 再現性

Trajectory = namedtuple(
    "Transition", ("state", "action", "reward", "done", "next_state")
)


class A2C:
    def __init__(
        self,
        device,
        n_actions,
        n_actionRange,
        n_features,
        learning_rate=0.01,
        gamma=0.9,
        tau=0.001,
        mSize=10000,
        batchSize=200,
    ):
        self.device = device
        self.n_actionRange = torch.tensor(list(n_actionRange))
        self.actorCriticEval = ActorCriticNet(n_actions, n_features).to(self.device)
        self.actorCriticTarget = ActorCriticNet(n_actions, n_features).to(self.device)
        print(self.device)
        print(self.actorCriticEval)
        print(self.actorCriticTarget)
        print("max action range:", self.n_actionRange[:, 0])

        self.memory = MemoryDataset(mSize)
        self.batchSize = batchSize

        self.lr = learning_rate
        # reward 衰減係數
        self.gamma = gamma
        self.tau = tau

        # optimizer 是訓練的工具
        # 傳入 net 的所有參數, 學習率
        self.optimizerCritic = torch.optim.Adam(
            self.actorCriticEval.critic.parameters(), lr=self.lr
        )
        self.optimizerActor = torch.optim.Adam(
            self.actorCriticEval.actor.parameters(), lr=self.lr
        )

    def choose_action(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        mean, std = self.actorCriticEval.action(state)
        # print(mean, std)
        action = torch.normal(mean, std)
        action = action * self.n_actionRange[:, 0].to(self.device)

        return action.cpu().detach().numpy()

    def store_trajectory(self, s, a, r, done, s_):
        self.memory.add(s, a, r, done, s_)

    # episode train
    def trainActor(self):
        if len(self.memory) < self.batchSize * 10:
            return

        batch = Trajectory(*zip(*self.memory.sample(self.batchSize)))

        # 轉成 np.array 加快轉換速度
        s = np.array(batch.state)
        # a = np.array(batch.action)
        # r = np.array(batch.reward)
        # done = np.array(batch.done)
        # s_ = np.array(batch.next_state)

        s = torch.FloatTensor(s).to(self.device)
        # a = torch.FloatTensor(batch.action)
        # r = torch.unsqueeze(torch.FloatTensor(batch.reward), dim=1)
        # done = torch.FloatTensor(batch.done)
        # s_ = torch.FloatTensor(batch.next_state)

        mean, std = self.actorCriticEval.action(s)
        gauss = Normal(mean, std)
        a = gauss.rsample()
        qVal = self.actorCriticEval.qValue(s, a)
        loss = -qVal.sum()

        self.optimizerActor.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizerActor.step()

        # print(loss.item())
        # print(list(self.actorCriticEval.actor.parameters()))
        # print("=============================================")

    # step train
    def trainCriticTD(self):
        if len(self.memory) < self.batchSize * 10:
            return

        batch = Trajectory(*zip(*self.memory.sample(self.batchSize)))

        # 轉成 np.array 加快轉換速度
        s = np.array(batch.state)
        a = np.array(batch.action)
        r = np.array(batch.reward)
        done = np.array(batch.done)
        s_ = np.array(batch.next_state)

        s = torch.FloatTensor(s).to(self.device)
        a = torch.FloatTensor(a).to(self.device)
        r = torch.FloatTensor(r).to(self.device)
        done = torch.FloatTensor(done).to(self.device)
        s_ = torch.FloatTensor(s_).to(self.device)

        mean, std = self.actorCriticTarget.action(s_)
        a_ = torch.normal(mean, std) * self.n_actionRange[:, 0].to(self.device)

        futureVal = torch.squeeze(self.actorCriticTarget.qValue(s_, a_))
        val = r + self.gamma * futureVal * (1 - done)
        target = val.detach()
        predict = torch.squeeze(self.actorCriticEval.qValue(s, a))

        self.optimizerCritic.zero_grad()
        loss = F.smooth_l1_loss(predict, target)
        loss.backward()
        self.optimizerCritic.step()

        # print(list(self.actorCriticEval.critic.parameters()))
        # print("=============================================")

    # 逐步更新 target NN
    def updateTarget(self):
        for paramEval, paramTarget in zip(
            self.actorCriticEval.parameters(), self.actorCriticTarget.parameters()
        ):
            paramTarget.data = paramEval.data + self.tau * (
                paramTarget.data - paramEval.data
            )


class ActorNet(torch.nn.Module):
    def __init__(self, n_actions, n_features):
        super(ActorNet, self).__init__()
        # 定義每層用什麼樣的形式
        self.fcMean1 = torch.nn.Linear(n_features, 5)
        self.fcMean2 = torch.nn.Linear(5, 3)
        self.fcMean3 = torch.nn.Linear(3, n_actions)

        self.fcStd1 = torch.nn.Linear(n_features, 5)
        self.fcStd2 = torch.nn.Linear(5, 3)
        self.fcStd3 = torch.nn.Linear(3, n_actions)

    def forward(self, x):  # 這同時也是 Module 中的 forward 功能
        # 正向傳播輸入值, 神經網絡分析出輸出值
        x_m = F.relu(self.fcMean1(x))
        x_m = F.relu(self.fcMean2(x_m))
        mean = self.fcMean3(x_m)

        x_s = F.relu(self.fcStd1(x))
        x_s = F.relu(self.fcStd2(x_s))
        # 加入 1e-14 防止 std = 0
        std = F.relu(self.fcStd3(x_s)) + 1e-14

        return mean, std


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

    def forward(self, x, a):  # 這同時也是 Module 中的 forward 功能
        # 正向傳播輸入值, 神經網絡分析出輸出值
        mean, std = self.actor(x)
        qVal = self.critic(x, a)

        return mean, std, qVal

    def action(self, x):
        mean, std = self.actor(x)
        return mean, std

    def qValue(self, x, a):
        qVal = self.critic(x, a)

        return qVal
