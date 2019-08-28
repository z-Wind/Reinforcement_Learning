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


class A2CBase(IModels):
    def __init__(
        self,
        device,
        #
        actorCriticEval,
        actorCriticTarget,
        optimizerCritic,
        optimizerActor,
        #
        n_actions,
        #
        learning_rate,
        gamma,
        tau,
        #
        updateTargetFreq,
        #
        mSize,
        batchSize,
        #
        transforms,
    ):
        """
        gamma: reward 的衰減係數
        tau: update target 的係數，target = eval + tau * (target - eval)
        updateTargetFreq: 更新 target 的頻率
        """
        super().__init__(device, mSize, transforms)

        self.device = device

        self.actorCriticEval = actorCriticEval.to(self.device)
        self.actorCriticTarget = actorCriticTarget.to(self.device)

        print("device", self.device)
        print(self.actorCriticEval)

        # self.memory = MemoryDataset(mSize, transforms)
        self.batchSize = batchSize

        self.lr = learning_rate
        # reward 衰減係數
        self.gamma = gamma
        # target = eval + tau * (target - eval)
        self.tau = tau

        self.updateTargetFreq = updateTargetFreq

        # optimizer 是訓練的工具
        self.optimizerCritic = optimizerCritic
        self.optimizerActor = optimizerActor

        # 從 1 開始，以免 updateTarget
        self.trainStep = 1

    def choose_action(self, state):
        state = torch.from_numpy(state).float()
        state = torch.unsqueeze(state, dim=0).to(self.device)
        action = self.actorCriticEval.action(state)

        log_prob = None

        return (action, log_prob)

    def store_trajectory(self, s, a, r, done, s_):
        self.memory.add(s, a, r, done, s_)

    def train_step(self):
        # 至少兩個，可讓 train_criticTD 順利動作，而無需加入額外判斷
        if len(self.memory) < 2:
            return

        self.train_criticTD()
        self.train_actor()

        self.trainStep += 1

        self.update_target()

    def train_episode(self):
        # 因 on policy，需清掉每一次 episode 的資料
        # 畢竟 actor 會隨時間更新，導致 critic 無法再使用舊資料訓練
        # 因 critic 訓練的是 V_pi(s)，所以無法得知當下 actor 在舊資料會使用的 action
        # 若是 QValue 則無此困擾，因 Q(s,a)，含有評估在 s 下執行 a 的價值
        del self.memory[:]

    def train_actor(self):
        # 取最近一筆訓練
        batch = Trajectory(*self.memory[-1])

        # 轉成 np.array 加快轉換速度
        s = np.array(batch.state)
        r = np.array(batch.reward)
        s_ = np.array(batch.next_state)

        s = torch.FloatTensor(s).to(self.device)
        s = torch.unsqueeze(s, dim=0)
        r = torch.FloatTensor(r).to(self.device)
        s_ = torch.FloatTensor(s_).to(self.device)
        s_ = torch.unsqueeze(s_, dim=0)

        nowVal = self.actorCriticTarget.criticism(s)
        nowVal = torch.squeeze(nowVal)

        futureVal = self.actorCriticTarget.criticism(s_)
        futureVal = torch.squeeze(futureVal)

        # advance = Q(s,a) - V(s) = r + gamma * V(s_) - V(s)
        adv = r + self.gamma * futureVal - nowVal
        adv = adv.detach()

        log_prob = batch.action[1]

        loss = -log_prob * adv
        loss = loss.mean()

        self.optimizerActor.zero_grad()
        loss.backward()
        self.optimizerActor.step()

        # print(loss.item())
        # print(list(self.actorCriticEval.actor.parameters()))
        # print("=============================================")

        return loss

    def train_criticTD(self):
        batch = Trajectory(*zip(*self.memory.sample(self.batchSize)))

        # 轉成 np.array 加快轉換速度
        s = np.array(batch.state)
        r = np.array(batch.reward)
        done = np.array(batch.done)
        s_ = np.array(batch.next_state)

        s = torch.FloatTensor(s).to(self.device)
        r = torch.FloatTensor(r).to(self.device)
        done = torch.FloatTensor(done).to(self.device)
        s_ = torch.FloatTensor(s_).to(self.device)

        futureVal = self.actorCriticTarget.criticism(s_)
        futureVal = torch.squeeze(futureVal)

        val = r + self.gamma * futureVal * (1 - done)
        target = val.detach()

        predict = self.actorCriticEval.criticism(s)
        predict = torch.squeeze(predict)

        loss_fn = torch.nn.MSELoss()
        self.optimizerCritic.zero_grad()
        loss = loss_fn(predict, target)
        loss.backward()
        self.optimizerCritic.step()

        # print(list(self.actorCriticEval.critic.parameters()))
        # print("=============================================")

        return loss

    # 逐步更新 target net
    def update_target(self):
        if (self.trainStep % self.updateTargetFreq) != 0:
            return

        print(f"Update target network... tau={self.tau}")
        for paramEval, paramTarget in zip(
            self.actorCriticEval.parameters(), self.actorCriticTarget.parameters()
        ):
            paramTarget.data = paramEval.data + self.tau * (
                paramTarget.data - paramEval.data
            )

    # def get_default_params_path(self, filePath, envName):
    #    _dirPath = os.path.dirname(os.path.realpath(filePath))
    #    paramsPath = os.path.join(
    #        _dirPath, f"params_{envName}_{type(self).__name__}_{self.device.type}.pkl"
    #    )

    #    return paramsPath

    def save_models(self, paramsPath):
        print(f"Save parameters to {paramsPath}")
        torch.save(self.actorCriticEval.state_dict(), paramsPath)

    def load_models(self, paramsPath):
        if not os.path.exists(paramsPath):
            return False

        print(f"Load parameters from {paramsPath}")
        self.actorCriticEval.load_state_dict(
            torch.load(paramsPath, map_location=self.device)
        )
        self.actorCriticTarget.load_state_dict(
            torch.load(paramsPath, map_location=self.device)
        )
        return True

    def eval(self):
        print("Evaluation Mode")
        self.actorCriticEval.eval()

    def print_info(self):
        print(f"totalTrainStep:{self.trainStep:7d}")
