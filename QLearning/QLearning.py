import numpy as np
import torch

torch.manual_seed(500)  # 固定隨機種子 for 再現性


class QLearning:
    def __init__(self, n_features, n_actions, learning_rate=0.01):
        self.n_actions = n_actions
        self.n_features = n_features
        self.net = Net(n_features, n_actions)
        print(self.net)

        self.lr = learning_rate
        # Q 衰減係數
        self.gamma = 1

        # optimizer 是訓練的工具
        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=self.lr
        )  # 傳入 net 的所有參數, 學習率
        # loss function
        self.lossFun = torch.nn.MSELoss()

    def choose_action(self, state):
        state = torch.from_numpy(state).float()
        value = self.net(state)
        action_max_value, action = torch.max(value, 0)

        if np.random.random() >= 0.95:  # epslion greedy
            action = np.random.choice(range(self.n_actions), 1)

        self.q = value[action.item()]

        return action.item()

    def store_trajectory(self, s, a, r, s_):
        self.nextState = torch.from_numpy(s_).float()
        self.reward = r

    def train(self):
        target = self.reward + self.gamma * torch.max(self.net(self.nextState))

        self.optimizer.zero_grad()
        loss = self.lossFun(target, self.q)
        loss.backward()
        self.optimizer.step()


class Net(torch.nn.Module):
    def __init__(self, n_features, n_actions):
        super(Net, self).__init__()
        # 定義每層用什麼樣的形式
        self.fc1 = torch.nn.Linear(n_features, 128)
        self.fc2 = torch.nn.Linear(128, n_actions)  # Prob of Left

    def forward(self, x):  # 這同時也是 Module 中的 forward 功能
        # 正向傳播輸入值, 神經網絡分析出輸出值
        model = torch.nn.Sequential(self.fc1, torch.nn.ReLU(), self.fc2)
        return model(x)
