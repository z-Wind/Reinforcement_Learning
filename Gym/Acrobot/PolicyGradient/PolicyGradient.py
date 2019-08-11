import numpy as np
import torch
from torch.distributions import Categorical

torch.manual_seed(500)  # 固定隨機種子 for 再現性


class PolicyGradient:
    def __init__(self, device, n_features, n_actions, learning_rate=0.01, gamma=0.99):
        self.device = device
        self.net = Net(n_features, n_actions).to(self.device)
        print(self.device)
        print(self.net)

        self.lr = learning_rate
        # reward 衰減係數
        self.gamma = gamma

        # optimizer 是訓練的工具
        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=self.lr
        )  # 傳入 net 的所有參數, 學習率

        self.saved_log_probs = []
        self.rewards = []
        self.eps = np.finfo(np.float32).eps.item()

    def choose_action(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        probs = self.net(state)

        m = Categorical(probs)
        action = m.sample()

        log_prob = m.log_prob(action)
        self.saved_log_probs.append(log_prob)
        # 也可自定，但因計算誤差，需調整 learning rate 才能學到東西
        # 要小心節點關係不變，不然往上更新會有問題
        # log_prob_m = torch.log(probs[action.item()])
        # self.saved_log_probs.append(log_prob_m)

        return action.item()

    def store_trajectory(self, s, a, r):
        self.rewards.append(r)

    def train(self):
        R = 0
        policy_loss = []
        rewards = []

        # 現在的 reward 是由現在的 action 造成的，過去的 action 影響不大
        # 越未來的 reward，現在的 action 影響會越來越小
        # 若是看直接看 total reward 不太能區分出 action 的好壞，導致學習不好
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            rewards.insert(0, R)

        rewards = torch.tensor(rewards)
        # 正規化 reward 並加入 machine epsilon (self.eps) 以免除以 0
        rewards = (rewards - rewards.mean()) / (rewards.std() + self.eps)
        for log_prob, reward in zip(self.saved_log_probs, rewards):
            # 最大化，故加上負號
            policy_loss.append(-log_prob * reward)

        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()

        del self.rewards[:]
        del self.saved_log_probs[:]

        # print(policy_loss.item())
        # print(list(self.net.parameters()))
        # print("=============================================")


class Net(torch.nn.Module):
    def __init__(self, n_features, n_actions):
        super(Net, self).__init__()
        # 定義每層用什麼樣的形式
        self.fc1 = torch.nn.Linear(n_features, 100)
        self.fc2 = torch.nn.Linear(100, n_actions)  # Prob of Left

    def forward(self, x):  # 這同時也是 Module 中的 forward 功能
        # 正向傳播輸入值, 神經網絡分析出輸出值
        model = torch.nn.Sequential(
            self.fc1, torch.nn.ReLU(), self.fc2, torch.nn.Softmax(dim=0)
        )
        return model(x)
