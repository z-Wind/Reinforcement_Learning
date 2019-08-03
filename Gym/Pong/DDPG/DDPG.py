import torch
import torch.nn.functional as F
from utils import MemoryDataset, OrnsteinUhlenbeckNoise
from collections import namedtuple

torch.manual_seed(500)  # 固定隨機種子 for 再現性

Trajectory = namedtuple(
    "Transition", ("state", "action", "reward", "done", "next_state")
)


class DDPG:
    def __init__(
        self,
        device,
        n_actions,
        n_actionRange,
        n_features,
        img_shape,
        learning_rate=0.01,
        gamma=0.9,
        tau=0.001,
        mSize=10000,
        batchSize=200,
        transforms=None,
    ):
        self.device = device
        self.max_noise_action = n_actionRange[0]

        self.n_actionRange = torch.tensor(list(n_actionRange))
        self.actorCriticEval = ActorCriticNet(img_shape, n_actions, n_features).to(
            self.device
        )
        self.actorCriticTarget = ActorCriticNet(img_shape, n_actions, n_features).to(
            self.device
        )
        print(self.device)
        print(self.actorCriticEval)
        print(self.actorCriticTarget)

        self.transforms = transforms
        self.memory = MemoryDataset(mSize, transforms=transforms)
        self.batchSize = batchSize

        self.lr = learning_rate
        # reward 衰減係數
        self.gamma = gamma
        self.tau = tau

        self.noise = OrnsteinUhlenbeckNoise(n_actions)

        # optimizer 是訓練的工具
        # 傳入 net 的所有參數, 學習率
        self.optimizerCritic = torch.optim.Adam(
            [
                {"params": self.actorCriticEval.img_features.parameters(), "lr": 0.01},
                {"params": self.actorCriticEval.critic.parameters()},
            ],
            lr=self.lr,
        )
        self.optimizerActor = torch.optim.Adam(
            [
                {"params": self.actorCriticEval.img_features.parameters(), "lr": 0.01},
                {"params": self.actorCriticEval.actor.parameters()},
            ],
            lr=self.lr,
        )

    def choose_action(self, state, n=0):
        state = torch.unsqueeze(self.transforms(state), dim=0).to(self.device)

        if not self.actorCriticEval.training:
            action = self.get_exploitation_action(state)
        elif n % 2 == 0:
            # validate every 5th episode
            action = self.get_exploitation_action(state)
        else:
            # get action based on observation, use exploration policy here
            action = self.get_exploration_action(state)

        return action.cpu().data.numpy()

    def get_exploitation_action(self, state):
        action = self.actorCriticEval.action(state)

        return action

    def get_exploration_action(self, state):
        action = self.actorCriticEval.action(state)
        noise = self.noise() * self.max_noise_action

        action = action + torch.from_numpy(noise).float().to(self.device)

        return action

    def store_trajectory(self, s, a, r, done, s_):
        self.memory.add(s, a, r, done, s_)

    # episode train
    def trainActor(self):
        if len(self.memory) < self.batchSize * 10:
            return

        batch = Trajectory(*zip(*self.memory.sample(self.batchSize)))

        s = [self.transforms(s) for s in batch.state]
        s = torch.stack(s).to(self.device)
        # a = torch.FloatTensor(batch.action)
        # a = torch.squeeze(a, dim=1)
        # r = torch.FloatTensor(batch.reward)
        # done = torch.FloatTensor(batch.done)
        # s_ = [self.transforms(s) for s in batch.next_state]
        # s_ = torch.stack(s_)

        a = self.actorCriticEval.action(s)
        qVal = self.actorCriticEval.qValue(s, a)
        loss = -qVal.mean()

        self.optimizerActor.zero_grad()
        loss.backward()
        self.optimizerActor.step()

        # print(loss.item())
        # print(list(self.actorCriticEval.img_features.parameters()))
        # print(list(self.actorCriticEval.actor.parameters()))
        # print("=============================================")

    # step train
    def trainCriticTD(self):
        if len(self.memory) < self.batchSize * 10:
            return

        batch = Trajectory(*zip(*self.memory.sample(self.batchSize)))

        s = [self.transforms(s) for s in batch.state]
        s = torch.stack(s).to(self.device)
        a = torch.FloatTensor(batch.action)
        a = torch.squeeze(a, dim=1).to(self.device)
        r = torch.FloatTensor(batch.reward).to(self.device)
        done = torch.FloatTensor(batch.done).to(self.device)
        s_ = [self.transforms(s) for s in batch.next_state]
        s_ = torch.stack(s_).to(self.device)

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

        # print(list(self.actorCriticEval.img_features.parameters()))
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
        self.fc1 = torch.nn.Linear(n_features, 128)
        self.fc2 = torch.nn.Linear(128, n_actions)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.softmax(self.fc2(x), dim=1)

        return x


class CriticNet(torch.nn.Module):
    def __init__(self, n_actions, n_features):
        super(CriticNet, self).__init__()
        # 定義每層用什麼樣的形式
        self.fcVal1_s = torch.nn.Linear(n_features, 1024)
        self.fcVal2_s = torch.nn.Linear(1024, 512)

        self.fcVal1_a = torch.nn.Linear(n_actions, 512)

        self.fcVal1 = torch.nn.Linear(1024, 512)
        self.fcVal2 = torch.nn.Linear(512, 1)

    def forward(self, x, a):  # 這同時也是 Module 中的 forward 功能
        # 正向傳播輸入值, 神經網絡分析出輸出值
        x_v = F.relu(self.fcVal1_s(x))
        x_v = F.relu(self.fcVal2_s(x_v))

        x_a = F.relu(self.fcVal1_a(a))

        x = torch.cat((x_v, x_a), dim=1)
        x = F.relu(self.fcVal1(x))
        qVal = self.fcVal2(x)

        return qVal


class CNN(torch.nn.Module):
    def __init__(self, img_shape, n_features):  # chan x h x w
        super(CNN, self).__init__()
        in_channels = img_shape[2]
        h = img_shape[0]
        w = img_shape[1]

        self.conv1 = torch.nn.Conv2d(in_channels, 32, 5)  # 32 x h-2 x w-2
        h -= 4
        w -= 4

        self.pool1 = torch.nn.MaxPool2d(2)  # 32 x (h-2)//2 x (w-2)//2
        h //= 2
        w //= 2

        self.conv2 = torch.nn.Conv2d(32, 64, 5)  # 64 x (h-2)//2-2 x (w-2)//2-2
        h -= 4
        w -= 4

        self.pool2 = torch.nn.MaxPool2d(2)  # 64 x ((h-2)//2-2)//2 x ((w-2)//2-2)//2
        h //= 2
        w //= 2

        self.fc1 = torch.nn.Linear(64 * h * w, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, n_features)

        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


class ActorCriticNet(torch.nn.Module):
    def __init__(self, img_shape, n_actions, n_features):
        super(ActorCriticNet, self).__init__()
        # 定義每層用什麼樣的形式
        self.img_features = CNN(img_shape, n_features)
        self.actor = ActorNet(n_actions, n_features)
        self.critic = CriticNet(n_actions, n_features)

    def forward(self, x, a):  # 這同時也是 Module 中的 forward 功能
        # 正向傳播輸入值, 神經網絡分析出輸出值
        x = self.img_features(x)
        action = self.actor(x)
        qVal = self.critic(x, a)

        return action, qVal

    def action(self, x):
        x = self.img_features(x)
        action = self.actor(x)

        return action

    def qValue(self, x, a):
        x = self.img_features(x)
        qVal = self.critic(x, a)

        return qVal
