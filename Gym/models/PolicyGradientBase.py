import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import namedtuple
import numpy as np
import os

from Gym.tools.utils import MemoryDataset
from .base import IModels

Trajectory = namedtuple(
    "Transition", ("state", "action", "reward", "done", "next_state")
)


class PolicyGradientBase(IModels):
    def __init__(
        self,
        device,
        #
        net,
        optimizer,
        #
        n_actions,
        #
        learning_rate,
        gamma,
        #
        mSize,
        #
        transforms,
    ):
        """
        gamma: reward 的衰減係數
        """
        super().__init__(device, mSize, transforms)

        self.device = device

        self.net = net.to(self.device)

        print("device", self.device)
        print(self.net)

        # self.memory = MemoryDataset(mSize, transforms)

        self.lr = learning_rate
        # reward 衰減係數
        self.gamma = gamma

        # optimizer 是訓練的工具
        self.optimizer = optimizer

    def choose_action(self, state):
        state = torch.from_numpy(state).float()
        state = torch.unsqueeze(state, dim=0).to(self.device)

        action = self.net(state)

        log_prob = None

        return (action, log_prob)

    def store_trajectory(self, s, a, r, done, s_):
        self.memory.add(s, a, r, done, s_)

    def train_step(self):
        pass

    def train_episode(self):
        self.train()

        # 因 on policy，需清掉每一次 episode 的資料
        # 畢竟 actor 會隨時間更新，導致 critic 無法再使用舊資料訓練
        # 因 critic 訓練的是 V_pi(s)，所以無法得知當下 actor 會使用的 action
        # 若是 QValue 則無此困擾，因 Q(s,a)，含有評估在 s 下執行 a 的價值
        del self.memory[:]

    def rewardCalFunc(self, rewards):
        R = 0
        result = []

        # 現在的 reward 是由現在的 action 造成的，過去的 action 影響不大
        # 越未來的 reward，現在的 action 影響會越來越小
        # 若是看直接看 total reward 不太能區分出 action 的好壞，導致學習不好
        for r in rewards[::-1]:
            R = r + self.gamma * R
            result.insert(0, R)

        return np.array(result)

    def train(self):
        batch = Trajectory(*zip(*self.memory))

        # 轉成 np.array 加快轉換速度
        r = np.array(batch.reward)
        log_prob = [a[1] for a in batch.action]

        r = self.rewardCalFunc(r)
        r = torch.FloatTensor(r).to(self.device)
        log_prob = torch.stack(log_prob)
        log_prob = torch.squeeze(log_prob, dim=1).to(self.device)

        r = r.detach()
        loss = -log_prob * r
        loss = loss.sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # print(loss.item())
        # print(list(self.net.parameters()))
        # print("=============================================")

        return loss

    # def get_default_params_path(self, filePath, envName):
    #    _dirPath = os.path.dirname(os.path.realpath(filePath))
    #    paramsPath = os.path.join(
    #        _dirPath, f"params_{envName}_{type(self).__name__}_{self.device.type}.pkl"
    #    )

    #    return paramsPath

    def save_models(self, paramsPath):
        print(f"Save parameters to {paramsPath}")
        torch.save(self.net.state_dict(), paramsPath)

    def load_models(self, paramsPath):
        if not os.path.exists(paramsPath):
            return False

        print(f"Load parameters from {paramsPath}")
        self.net.load_state_dict(torch.load(paramsPath, map_location=self.device))
        self.net.load_state_dict(torch.load(paramsPath, map_location=self.device))

        return True

    def eval(self):
        print("Evaluation Mode")
        self.net.eval()

    def print_info(self):
        print()
