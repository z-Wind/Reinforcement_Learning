import numpy as np
import torch
from torch.distributions import Categorical

torch.manual_seed(500)  # 固定隨機種子 for 再現性


class A2C:
    def __init__(self, n_actions, n_features, learning_rate=0.01):
        self.actor = ActorNet(n_actions, n_features)
        self.critic = CriticNet(n_features)
        print(self.actor)
        print(self.critic)

        self.lr = learning_rate
        # reward 衰減係數
        self.gamma = 0.9

        # optimizer 是訓練的工具
        # 傳入 net 的所有參數, 學習率
        self.optimizerActor = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.optimizerCritic = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        self.saved_log_probs = []
        self.rewards = []
        self.states = []

    def choose_action(self, state):
        state = torch.from_numpy(state).float()
        probs = self.actor(state)

        m = Categorical(probs)
        action = m.sample()
        
        log_prob = m.log_prob(action)
        self.saved_log_probs.append(log_prob)
        # 也可自定，但因計算誤差，需調整 learning rate 才能學到東西
        # 要小心節點關係不變，不然往上更新會有問題
        # log_prob_m = torch.log(probs[action.item()])
        # self.saved_log_probs.append(log_prob_m)

        return action.item()

    def store_trajectory(self, s, a, r, s_):
        self.rewards.append(r)
        self.states.append(s)
        self.nextState = s_

    # episode train
    def trainActor(self):
        R = 0
        policy_loss = []
        rewards = []

        # 現在的 reward 是由現在的 action 造成的，過去的 action 影響不大
        # 越未來的 reward，現在的 action 影響會越來越小
        # 若是看直接看 total reward 不太能區分出 action 的好壞，導致學習不好
        nextStates = self.states + [self.nextState]
        for r, s, s_ in zip(self.rewards[::-1], self.states[::-1], nextStates[::-1]):
            R_now = (
                r
                + self.critic(torch.tensor(s_).float())
                - self.critic(torch.tensor(s).float())
            )
            R = R_now + self.gamma * R
            rewards.insert(0, R)

        for log_prob, reward in zip(self.saved_log_probs, rewards):
            # 最大化，故加上負號
            policy_loss.append(-log_prob * reward)

        # Actor
        self.optimizerActor.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        self.optimizerActor.step()

        del self.rewards[:]
        del self.saved_log_probs[:]

    # step train
    def trainCritic(self):
        reward = self.rewards[-1]
        # TD method
        target = reward + self.gamma * self.critic(
            torch.from_numpy(self.nextState).float()
        )

        self.optimizerCritic.zero_grad()
        lossFun = torch.nn.MSELoss()
        loss = lossFun(target, self.critic(torch.from_numpy(self.states[-1]).float()))
        loss.backward()
        self.optimizerCritic.step()


class ActorNet(torch.nn.Module):
    def __init__(self, n_actions, n_features):
        super(ActorNet, self).__init__()
        # 定義每層用什麼樣的形式
        self.fc1 = torch.nn.Linear(n_features, 128)
        self.fc2 = torch.nn.Linear(128, n_actions)  # Prob of Left

    def forward(self, x):  # 這同時也是 Module 中的 forward 功能
        # 正向傳播輸入值, 神經網絡分析出輸出值
        model = torch.nn.Sequential(
            self.fc1, torch.nn.ReLU(), self.fc2, torch.nn.Softmax()
        )
        return model(x)


class CriticNet(torch.nn.Module):
    def __init__(self, n_features):
        super(CriticNet, self).__init__()
        # 定義每層用什麼樣的形式
        self.fc1 = torch.nn.Linear(n_features, 128)
        self.fc2 = torch.nn.Linear(128, 1)  # Prob of Left

    def forward(self, x):  # 這同時也是 Module 中的 forward 功能
        # 正向傳播輸入值, 神經網絡分析出輸出值
        model = torch.nn.Sequential(self.fc1, torch.nn.ReLU(), self.fc2)
        return model(x)
