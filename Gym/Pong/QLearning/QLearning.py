import numpy as np
import torch
import torch.nn.functional as F
from .utils import MemoryDataset
from collections import namedtuple

torch.manual_seed(500)  # 固定隨機種子 for 再現性

Trajectory = namedtuple(
    "Transition", ("state", "action", "reward", "done", "next_state")
)


class QLearning:
    def __init__(
        self,
        device,
        n_actions,
        img_shape,
        learning_rate=0.0001,
        gamma=0.99,
        tau=0.0,
        epsilonStart=0,
        epsilonUpdateSteps=10 ** 5,
        updateTargetFreq=10000,
        mSize=1_000_000,
        batchSize=32,
        startTrainLen=100,
        transforms=None,
    ):
        self.device = device
        self.n_actions = n_actions
        self.img_shape = img_shape
        self.net = Net(img_shape, n_actions).to(self.device)
        self.netTarget = Net(img_shape, n_actions).to(self.device)
        print("device", self.device)
        print(self.net)
        print(self.netTarget)

        self.lr = learning_rate
        # Q 衰減係數
        self.gamma = gamma
        self.tau = tau

        self.epsilon = epsilonStart
        self.epsilonStart = epsilonStart
        self.epsilonUpdateSteps = epsilonUpdateSteps

        self.updateTargetFreq = updateTargetFreq

        # optimizer 是訓練的工具
        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=self.lr
        )  # 傳入 net 的所有參數, 學習率
        # self.optimizer = torch.optim.RMSprop(self.net.parameters(), lr=self.lr)

        # loss function
        self.lossFun = torch.nn.MSELoss()

        self.transforms = transforms
        self.memory = MemoryDataset(mSize, transforms=transforms)
        self.batchSize = batchSize
        self.startTrainLen = startTrainLen

        # 從 1 開始，以免 updateTarget
        self.trainStep = 1

    def decay_epsilon(self):
        self.epsilon = min(
            0.98, self.epsilonStart + self.trainStep / self.epsilonUpdateSteps
        )

    def choose_action(self, state):
        choice = np.random.choice([0, 1], p=((1 - self.epsilon), self.epsilon))

        # epslion greedy
        if choice == 0 and self.net.training:
            action = np.random.choice(range(self.n_actions), 1)
        else:
            # state = torch.unsqueeze(self.transforms(state), dim=0).to(self.device)
            state = torch.FloatTensor(state)
            state = torch.unsqueeze(state, dim=0).to(self.device)
            qValues = self.net(state)
            action_max_value, action = torch.max(qValues, 1)

        return action.item()

    def store_trajectory(self, s, a, r, done, s_):
        self.memory.add(s, a, r, done, s_)

    def train(self):
        if len(self.memory) < self.startTrainLen:
            return
        self.trainStep += 1

        batch = Trajectory(*zip(*self.memory.sample(self.batchSize)))

        # 轉成 np.array 加快轉換速度
        s = np.array(batch.state)
        a = np.array(batch.action)
        r = np.array(batch.reward)
        done = np.array(batch.done)
        s_ = np.array(batch.next_state)

        s = torch.FloatTensor(s).to(self.device)
        a = torch.LongTensor(a)
        a = torch.unsqueeze(a, 1).to(self.device)
        r = torch.FloatTensor(r).to(self.device)
        done = torch.FloatTensor(done).to(self.device)
        s_ = torch.FloatTensor(s_).to(self.device)

        # 在 dim=1，以 a 為 index 取值
        qValue = self.net(s).gather(1, a).squeeze(1)

        # caculate Target
        qNext = self.netTarget(s_).detach()  # detach from graph, don't backpropagate
        # done 是關鍵之一，不導入計算會導致 qNext 預估錯誤
        # 這也是讓 qValue 收斂的要素，不然 target 會一直往上累加，進而估不準
        target = r + self.gamma * qNext.max(1)[0] * (1 - done)

        self.optimizer.zero_grad()
        loss = self.lossFun(qValue, target.detach())
        loss.backward()
        # torch.nn.utils.clip_grad_norm(self.net.parameters(), 0.5)
        self.optimizer.step()

        # print(list(self.net.parameters()))

    # 逐步更新 target NN, 必須一定的訓練後，再更新
    def updateTarget(self):
        if (self.trainStep % self.updateTargetFreq) != 0:
            return

        print(f"Update target network... tau={self.tau}")
        for paramEval, paramTarget in zip(
            self.net.parameters(), self.netTarget.parameters()
        ):
            paramTarget.data = paramEval.data + self.tau * (
                paramTarget.data - paramEval.data
            )


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
