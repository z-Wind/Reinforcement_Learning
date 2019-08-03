import numpy as np
import torch
import torch.nn.functional as F
from utils import MemoryDataset
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
        learning_rate=0.01,
        gamma=0.9,
        tau=0.001,
        mSize=10000,
        batchSize=200,
        transforms=None,
    ):
        self.device = device
        self.n_actions = n_actions
        self.img_shape = img_shape
        self.net = Net(img_shape, n_actions).to(self.device)
        print(self.device)
        print(self.net)

        self.lr = learning_rate
        # Q 衰減係數
        self.gamma = gamma
        self.tau = tau

        # optimizer 是訓練的工具
        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=self.lr
        )  # 傳入 net 的所有參數, 學習率
        # loss function
        self.lossFun = torch.nn.MSELoss()

        self.transforms = transforms
        self.memory = MemoryDataset(mSize, transforms=transforms)
        self.batchSize = batchSize

    def choose_action(self, state):
        state = torch.unsqueeze(self.transforms(state), dim=0).to(self.device)
        value = self.net(state)
        action_max_value, action = torch.max(value, 1)

        if np.random.random() >= 0.95:  # epslion greedy
            action = np.random.choice(range(self.n_actions), 1)

        return action.item()

    def store_trajectory(self, s, a, r, done, s_):
        self.memory.add(s, a, r, done, s_)

    def train(self):
        if len(self.memory) < self.batchSize * 10:
            return

        batch = Trajectory(*zip(*self.memory.sample(self.batchSize)))

        s = [self.transforms(s) for s in batch.state]
        s = torch.stack(s).to(self.device)
        a = torch.LongTensor(batch.action)
        a = torch.unsqueeze(a, 1).to(self.device)
        r = torch.FloatTensor(batch.reward).to(self.device)
        done = torch.FloatTensor(batch.done).to(self.device)
        s_ = [self.transforms(s) for s in batch.next_state]
        s_ = torch.stack(s_).to(self.device)

        # 在 dim=1，以 a 為 index 取值
        qValue = self.net(s).gather(1, a).squeeze(1)
        qNext = self.net(s_).detach()  # detach from graph, don't backpropagate
        # done 是關鍵之一，不導入計算會導致 qNext 預估錯誤
        # 這也是讓 qValue 收斂的要素，不然 target 會一直往上累加，進而估不準
        target = r + self.gamma * qNext.max(1)[0] * (1 - done)

        self.optimizer.zero_grad()
        loss = self.lossFun(target.detach(), qValue)
        loss.backward()
        # torch.nn.utils.clip_grad_norm(self.net.parameters(), 0.5)
        self.optimizer.step()


class Net(torch.nn.Module):
    def __init__(self, img_shape, n_actions):
        super(Net, self).__init__()
        # 定義每層用什麼樣的形式
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
        self.fc3 = torch.nn.Linear(84, n_actions)

        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x):  # 這同時也是 Module 中的 forward 功能
        # 正向傳播輸入值, 神經網絡分析出輸出值
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
