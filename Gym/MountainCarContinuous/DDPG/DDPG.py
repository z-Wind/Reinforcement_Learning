import torch
import torch.nn.functional as F
from torch.distributions import Normal
from utils import MemoryDataset, OrnsteinUhlenbeckNoise
from collections import namedtuple

torch.manual_seed(500)  # 固定隨機種子 for 再現性

Trajectory = namedtuple(
    "Transition", ("state", "action", "reward", "done", "next_state")
)


class DDPG:
    def __init__(
        self,
        n_actions,
        n_actionRange,
        n_features,
        learning_rate=0.01,
        gamma=0.9,
        tau=0.001,
        mSize=10000,
        batchSize=200,
    ):
        self.n_actionRange = torch.tensor(list(n_actionRange))
        self.actorCriticEval = ActorCriticNet(
            n_actions, n_features, self.n_actionRange[:, 0]
        )
        self.actorCriticTarget = ActorCriticNet(
            n_actions, n_features, self.n_actionRange[:, 0]
        )
        print(self.actorCriticEval)
        print(self.actorCriticTarget)
        print("max action range:", self.n_actionRange[:, 0])

        self.memory = MemoryDataset(mSize)
        self.batchSize = batchSize

        self.lr = learning_rate
        # reward 衰減係數
        self.gamma = gamma
        self.tau = tau

        self.noise = OrnsteinUhlenbeckNoise(n_actions)

        # optimizer 是訓練的工具
        # 傳入 net 的所有參數, 學習率
        self.optimizerCritic = torch.optim.Adam(
            self.actorCriticEval.critic.parameters(), lr=self.lr
        )
        self.optimizerActor = torch.optim.Adam(
            self.actorCriticEval.actor.parameters(), lr=self.lr
        )

    def choose_action(self, state, n=0):
        if not self.actorCriticEval.training:
            return self.get_exploitation_action(state)

        if n % 50 == 0:
            # validate every 5th episode
            action = self.get_exploitation_action(state)
        else:
            # get action based on observation, use exploration policy here
            action = self.get_exploration_action(state)

        return action

    def get_exploitation_action(self, state):
        state = torch.from_numpy(state).float()
        action = self.actorCriticEval.action(state)

        return action.data.numpy()

    def get_exploration_action(self, state):
        state = torch.from_numpy(state).float()
        action = self.actorCriticEval.action(state)
        action = action.data.numpy() + (
            self.noise() * self.n_actionRange[:, 0].data.numpy()
        )

        return action

    def store_trajectory(self, s, a, r, done, s_):
        self.memory.add(s, a, r, done, s_)

    # episode train
    def trainActor(self):
        if len(self.memory) < self.batchSize * 10:
            return

        batch = Trajectory(*zip(*self.memory.sample(self.batchSize)))

        s = torch.FloatTensor(batch.state)
        # a = torch.FloatTensor(batch.action)
        # r = torch.unsqueeze(torch.FloatTensor(batch.reward), dim=1)
        # done = torch.FloatTensor(batch.done)
        # s_ = torch.FloatTensor(batch.next_state)

        a = self.actorCriticEval.action(s)
        qVal = self.actorCriticEval.qValue(s, a)
        loss = -qVal.mean()

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

        s = torch.FloatTensor(batch.state)
        a = torch.FloatTensor(batch.action)
        r = torch.FloatTensor(batch.reward)
        done = torch.FloatTensor(batch.done)
        s_ = torch.FloatTensor(batch.next_state)

        a_ = self.actorCriticTarget.action(s_)

        futureVal = torch.squeeze(self.actorCriticTarget.qValue(s_, a_))
        val = r + self.gamma * futureVal * (1 - done)
        target = val.detach()
        predict = torch.squeeze(self.actorCriticEval.qValue(s, a))

        loss_fn = torch.nn.MSELoss(reduction="sum")
        self.optimizerCritic.zero_grad()
        loss = loss_fn(target, predict)
        loss.backward()
        # 梯度裁剪，以免爆炸
        # torch.nn.utils.clip_grad_norm(actor_network.parameters(),0.5)
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
    def __init__(self, n_actions, n_features, action_lim):
        super(ActorNet, self).__init__()
        self.action_lim = action_lim
        # 定義每層用什麼樣的形式
        self.fc1 = torch.nn.Linear(n_features, 20)
        self.fc2 = torch.nn.Linear(20, 10)
        self.fc3 = torch.nn.Linear(10, 5)
        self.fc4 = torch.nn.Linear(5, n_actions)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        action = torch.tanh(self.fc4(x)) * self.action_lim

        return action


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
    def __init__(self, n_actions, n_features, action_lim):
        super(ActorCriticNet, self).__init__()
        # 定義每層用什麼樣的形式
        self.actor = ActorNet(n_actions, n_features, action_lim)
        self.critic = CriticNet(n_actions, n_features)

    def forward(self, x, a):  # 這同時也是 Module 中的 forward 功能
        # 正向傳播輸入值, 神經網絡分析出輸出值
        action = self.actor(x)
        qVal = self.critic(x, a)

        return action, qVal

    def action(self, x):
        action = self.actor(x)
        return action

    def qValue(self, x, a):
        qVal = self.critic(x, a)

        return qVal
