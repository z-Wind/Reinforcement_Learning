import numpy as np
import torch
from collections import namedtuple
import random

torch.manual_seed(500)  # 固定隨機種子 for 再現性

Trajectory = namedtuple(
    "Transition", ("state", "action", "reward", "done", "next_state")
)


# 很重要的機制，無此機制，比較難收斂
# 可以試著將 capacity & BATCH_SIZE 設為 1 看看
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a trajectory."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Trajectory(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class QLearning:
    def __init__(self, device, n_features, n_actions, learning_rate=0.01, gamma=0.9):
        self.device = device
        self.n_actions = n_actions
        self.n_features = n_features
        self.net = Net(n_features, n_actions).to(self.device)
        print(self.device)
        print(self.net)

        self.lr = learning_rate
        # Q 衰減係數
        self.gamma = gamma

        # optimizer 是訓練的工具
        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=self.lr
        )  # 傳入 net 的所有參數, 學習率
        # loss function
        self.lossFun = torch.nn.MSELoss()

        self.trajectories = ReplayMemory(10000)
        self.BATCH_SIZE = 50

    def choose_action(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        value = self.net(state)
        action_max_value, action = torch.max(value, 0)

        if np.random.random() >= 0.95:  # epslion greedy
            action = np.random.choice(range(self.n_actions), 1)

        return action.item()

    def store_trajectory(self, s, a, r, done, s_):
        self.trajectories.push(s, a, r, done, s_)

    def train(self):
        if len(self.trajectories) < self.BATCH_SIZE:
            return

        trajectories = self.trajectories.sample(self.BATCH_SIZE)
        batch = Trajectory(*zip(*trajectories))

        s = batch.state
        s = torch.tensor(s).float().to(self.device)
        a = batch.action
        a = torch.tensor(a).long()
        a = torch.unsqueeze(a, 1).to(self.device)  # 在 dim=1 增加維度 ex: (50,) => (50,1)
        r = batch.reward
        r = torch.tensor(r).float().to(self.device)
        done = batch.done
        done = torch.tensor(done).float().to(self.device)
        s_ = batch.next_state
        s_ = torch.tensor(s_).float().to(self.device)

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
    def __init__(self, n_features, n_actions):
        super(Net, self).__init__()
        # 定義每層用什麼樣的形式
        self.fc1 = torch.nn.Linear(n_features, 10)
        self.fc2 = torch.nn.Linear(10, n_actions)  # Prob of Left

    def forward(self, x):  # 這同時也是 Module 中的 forward 功能
        # 正向傳播輸入值, 神經網絡分析出輸出值
        model = torch.nn.Sequential(self.fc1, torch.nn.ReLU6(), self.fc2)
        return model(x)
