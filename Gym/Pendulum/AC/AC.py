import torch
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

from Gym.models.A2CBase import A2CBase, Trajectory


# actor-critic
class AC(A2CBase):
    def __init__(
        self,
        device,
        n_actions,
        n_features,
        max_actions,
        learning_rate=0.01,
        gamma=0.9,
        tau=0.001,
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
            updateTargetFreq=updateTargetFreq,
            mSize=mSize,
            batchSize=batchSize,
            transforms=transforms,
        )

        self.max_actions = max_actions
        self.startTrainSize = startTrainSize

    def choose_action(self, state):
        action, _ = super().choose_action(state)

        # std 可為負的，Normal 有處理
        m = Normal(action[:, 0], action[:, 1])
        a = m.sample()
        a = a.numpy()
        a = np.clip(a, -1, 1)
        a = a * self.max_actions

        action = action.cpu().data.numpy()

        return action, a

    def train_step(self):
        # 至少兩個，可讓 train_criticTD 順利動作，而無需加入額外判斷
        if len(self.memory) < self.startTrainSize:
            return

        self.train_criticTD()
        self.train_actor()

        self.trainStep += 1

        self.update_target()

    def train_episode(self):
        # 因是用 Q value 評估，故為 off policy
        # 覆蓋原本的 train_episode，取消刪除 memory
        # 若不取消，會一直學不好，因為概念上的不同
        pass

    # episode train
    def train_actor(self):
        batch = Trajectory(*zip(*self.memory.sample(self.batchSize)))

        # 轉成 np.array 加快轉換速度
        s = np.array(batch.state)

        s = torch.FloatTensor(s).to(self.device)

        _, qVal = self.actorCriticEval(s)
        loss = -qVal.sum()

        self.optimizerActor.zero_grad()
        loss.backward()
        self.optimizerActor.step()

        # print(loss.item())
        # print(list(self.actorCriticEval.actor.parameters()))
        # print("=============================================")

    # step train
    def train_criticTD(self):
        batch = Trajectory(*zip(*self.memory.sample(self.batchSize)))

        # 轉成 np.array 加快轉換速度
        s = np.array(batch.state)
        a = np.array(batch.action)
        r = np.array(batch.reward)
        done = np.array(batch.done)
        s_ = np.array(batch.next_state)

        s = torch.FloatTensor(s).to(self.device)
        a = torch.FloatTensor(a).to(self.device)
        a = torch.squeeze(a, dim=1)
        r = torch.FloatTensor(r).to(self.device)
        done = torch.FloatTensor(done).to(self.device)
        s_ = torch.FloatTensor(s_).to(self.device)

        _, futureVal = self.actorCriticTarget(s_)
        futureVal = torch.squeeze(futureVal)
        val = r + self.gamma * futureVal * (1 - done)
        target = val.detach()

        predict = torch.squeeze(self.actorCriticEval.criticism(s, a))

        self.optimizerCritic.zero_grad()
        loss = F.mse_loss(predict, target)
        loss.backward()
        self.optimizerCritic.step()

        # print(loss.item())
        # print(list(self.actorCriticEval.critic.parameters()))
        # print("=============================================")


# return mean, std
class ActorNet(torch.nn.Module):
    def __init__(self, n_actions, n_features):
        super(ActorNet, self).__init__()
        # 定義每層用什麼樣的形式
        self.fc1 = torch.nn.Linear(n_features, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))

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
