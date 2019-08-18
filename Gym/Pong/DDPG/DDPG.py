import torch
import torch.nn.functional as F
from collections import namedtuple
import numpy as np

from Gym.tools.utils import MemoryDataset, OrnsteinUhlenbeckNoise

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
        learning_rate=0.0001,
        gamma=0.99,
        tau=0.01,
        epsilonStart=0,
        epsilonUpdateSteps=10 ** 5,
        updateTargetFreq=10000,
        mSize=1_000_000,
        batchSize=32,
        startTrainSize=100,
        transforms=None,
    ):
        self.device = device
        self.max_noise_action = n_actionRange[0]
        self.max_noise_actionStart = self.max_noise_action

        self.n_actionRange = torch.tensor(list(n_actionRange))
        self.actorCriticEval = ActorCriticNet(img_shape, n_actions, n_features).to(
            self.device
        )
        self.actorCriticTarget = ActorCriticNet(img_shape, n_actions, n_features).to(
            self.device
        )

        print(self.device)
        print(self.actorCriticEval)

        self.transforms = transforms
        self.memory = MemoryDataset(mSize, transforms=transforms)
        self.batchSize = batchSize
        self.startTrainSize = startTrainSize

        self.lr = learning_rate
        # reward 衰減係數
        self.gamma = gamma
        self.tau = tau

        self.epsilon = epsilonStart
        self.epsilonStart = epsilonStart
        self.epsilonUpdateSteps = epsilonUpdateSteps

        self.updateTargetFreq = updateTargetFreq

        self.noise = OrnsteinUhlenbeckNoise(n_actions)

        # optimizer 是訓練的工具
        # 傳入 net 的所有參數, 學習率
        self.optimizerCritic = torch.optim.Adam(
            [
                {
                    "params": self.actorCriticEval.img_featuresCritic.parameters(),
                    "lr": self.lr,
                },
                {"params": self.actorCriticEval.critic.parameters()},
            ],
            lr=self.lr,
        )
        self.optimizerActor = torch.optim.Adam(
            [
                {
                    "params": self.actorCriticEval.img_featuresActor.parameters(),
                    "lr": self.lr,
                },
                {"params": self.actorCriticEval.actor.parameters()},
            ],
            lr=self.lr,
        )

        # 從 1 開始，以免 updateTarget
        self.trainStep = 1

    def decay_epsilon(self):
        self.epsilon = min(
            0.98, self.epsilonStart + self.trainStep / self.epsilonUpdateSteps
        )
        self.max_noise_action = max(
            1, self.max_noise_actionStart - self.trainStep / self.epsilonUpdateSteps
        )

    def choose_action(self, state):
        state = torch.FloatTensor(state)
        state = torch.unsqueeze(state, dim=0).to(self.device)

        # choice = np.random.choice([0, 1], p=((1 - self.epsilon), self.epsilon))

        # if choice == 0 and self.actorCriticEval.training:
        #    # get action based on observation, use exploration policy here
        #    action = self.get_exploration_action(state)
        # else:
        #    # validate every nth episode
        #    action = self.get_exploitation_action(state)

        if self.actorCriticEval.training:
            # get action based on observation, use exploration policy here
            action = self.get_exploration_action(state)
        else:
            # validate every nth episode
            action = self.get_exploitation_action(state)

        return action.cpu().data.numpy()

    def get_exploitation_action(self, state):
        action = self.actorCriticEval.action(state)

        return action

    def get_exploration_action(self, state):
        action = self.actorCriticEval.action(state)
        noise = self.noise() * self.max_noise_action

        action = action + torch.from_numpy(noise).float().to(self.device)
        action = F.softmax(action, dim=1)

        return action

    def store_trajectory(self, s, a, r, done, s_):
        self.memory.add(s, a, r, done, s_)

    def train(self):
        if len(self.memory) < self.startTrainSize:
            return

        self.trainStep += 1

        self.trainCriticTD()
        self.trainActor()

    # episode train
    def trainActor(self):
        batch = Trajectory(*zip(*self.memory.sample(self.batchSize)))

        # 轉成 np.array 加快轉換速度
        s = np.array(batch.state)
        s = torch.FloatTensor(s).to(self.device)

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
        batch = Trajectory(*zip(*self.memory.sample(self.batchSize)))

        # 轉成 np.array 加快轉換速度
        s = np.array(batch.state)
        a = np.array(batch.action)
        r = np.array(batch.reward)
        done = np.array(batch.done)
        s_ = np.array(batch.next_state)

        s = torch.FloatTensor(s).to(self.device)
        a = torch.FloatTensor(a)
        a = torch.squeeze(a, dim=1).to(self.device)
        r = torch.FloatTensor(r).to(self.device)
        done = torch.FloatTensor(done).to(self.device)
        s_ = torch.FloatTensor(s_).to(self.device)

        a_, futureVal = self.actorCriticTarget(s_)

        futureVal = torch.squeeze(futureVal)
        val = r + self.gamma * futureVal * (1 - done)
        target = val.detach()
        predict = self.actorCriticEval.qValue(s, a)
        predict = torch.squeeze(predict)

        loss_fn = torch.nn.MSELoss()
        self.optimizerCritic.zero_grad()
        loss = loss_fn(predict, target)
        loss.backward()
        # 梯度裁剪，以免爆炸
        # torch.nn.utils.clip_grad_norm(actor_network.parameters(),0.5)
        self.optimizerCritic.step()

        # print(list(self.actorCriticEval.img_features.parameters()))
        # print(list(self.actorCriticEval.critic.parameters()))
        # print("=============================================")

        return loss

    # 逐步更新 target NN
    def updateTarget(self):
        if (self.trainStep % self.updateTargetFreq) != 0:
            return

        print(f"Update target network... tau={self.tau}")
        for paramEval, paramTarget in zip(
            self.actorCriticEval.parameters(), self.actorCriticTarget.parameters()
        ):
            paramTarget.data = paramEval.data + self.tau * (
                paramTarget.data - paramEval.data
            )

    def teach(self):
        if len(self.memory) < self.batchSize * 10:
            return None

        batch = Trajectory(*zip(*self.memory.sample(self.batchSize)))

        # 轉成 np.array 加快轉換速度
        s = np.array(batch.state)
        a = np.array(batch.action)
        r = np.array(batch.reward)
        done = np.array(batch.done)
        s_ = np.array(batch.next_state)

        s = torch.FloatTensor(s).to(self.device)
        a = torch.FloatTensor(a)
        a = torch.squeeze(a, dim=1).to(self.device)
        r = torch.FloatTensor(r).to(self.device)
        done = torch.FloatTensor(done).to(self.device)
        s_ = torch.FloatTensor(s_).to(self.device)

        # Critic
        lossC = self.trainCriticTD()

        # Actor
        action = self.actorCriticEval.action(s)
        loss_fn = torch.nn.MSELoss()
        self.optimizerActor.zero_grad()
        lossA = loss_fn(a, action)
        lossA.backward()
        self.optimizerActor.step()

        loss = lossC + lossA
        # print(f"loss:{loss:.2f} lossC:{lossC:.2f} lossA:{lossA:.2f}")

        return loss.item()


class ActorNet(torch.nn.Module):
    def __init__(self, n_actions, n_features):
        super(ActorNet, self).__init__()
        # 定義每層用什麼樣的形式
        self.fc1 = torch.nn.Linear(n_features, n_actions)

    def forward(self, state):
        x = F.softmax(self.fc1(state), dim=1)

        return x


class CriticNet(torch.nn.Module):
    def __init__(self, n_actions, n_features):
        super(CriticNet, self).__init__()
        # 定義每層用什麼樣的形式
        self.fcVal1_s = torch.nn.Linear(n_features, 64)

        self.fcVal1_a = torch.nn.Linear(n_actions, 64)

        self.fcVal1 = torch.nn.Linear(128, 1)

    def forward(self, x, a):  # 這同時也是 Module 中的 forward 功能
        # 正向傳播輸入值, 神經網絡分析出輸出值
        x_v = F.relu(self.fcVal1_s(x))

        x_a = F.relu(self.fcVal1_a(a))

        x = torch.cat((x_v, x_a), dim=1)
        qVal = self.fcVal1(x)

        return qVal


class CNN(torch.nn.Module):
    def __init__(self, img_shape, n_features):  # chan x h x w
        super(CNN, self).__init__()
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

        # self.pool2 = torch.nn.MaxPool2d(2)  # 64 x ((h-2)//2-2)//2 x ((w-2)//2-2)//2
        # h //= 2
        # w //= 2

        kernel_size = 3
        stride = 1
        padding = 0
        self.conv3 = torch.nn.Conv2d(
            64, 64, kernel_size=kernel_size, stride=stride, padding=padding
        )
        h = (h + padding * 2 - kernel_size) // stride + 1
        w = (w + padding * 2 - kernel_size) // stride + 1

        self.fc1 = torch.nn.Linear(64 * h * w, 512)
        self.fc2 = torch.nn.Linear(512, n_features)

        # self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x):
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


class ActorCriticNet(torch.nn.Module):
    def __init__(self, img_shape, n_actions, n_features):
        super(ActorCriticNet, self).__init__()
        # 定義每層用什麼樣的形式
        self.img_featuresActor = CNN(img_shape, n_features)
        self.img_featuresCritic = CNN(img_shape, n_features)
        self.actor = ActorNet(n_actions, n_features)
        self.critic = CriticNet(n_actions, n_features)

    def forward(self, x):  # 這同時也是 Module 中的 forward 功能
        # 正向傳播輸入值, 神經網絡分析出輸出值
        x1 = self.img_featuresActor(x)
        action = self.actor(x1)

        x2 = self.img_featuresCritic(x)
        qVal = self.critic(x2, action)

        return action, qVal

    def action(self, x):
        x = self.img_featuresActor(x)
        action = self.actor(x)

        return action

    def qValue(self, x, a):
        x = self.img_featuresCritic(x)
        qVal = self.critic(x, a)

        return qVal
