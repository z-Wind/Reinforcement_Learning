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
        learning_rate=0.0001,
        gamma=0.99,
        tau=0.01,
        #
        updateTargetFreq=10000,
        #
        mSize=1_000_000,
        batchSize=32,
        #
        transforms=None,
    ):
        self.device = device

        self.actorCriticEval = actorCriticEval.to(self.device)
        self.actorCriticTarget = actorCriticTarget.to(self.device)

        print("device", self.device)
        print(self.actorCriticEval)

        self.memory = MemoryDataset(mSize, transforms)
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
        action = self.actorCriticEval.actor(state)

        log_prob = None

        return (action, log_prob)

    def store_trajectory(self, s, a, r, done, s_):
        self.memory.add(s, a, r, done, s_)

    def train_step(self):
        if len(self.memory) < 2:
            return

        self.trainStep += 1

        self.train_criticTD()
        self.train_actor()

        self.update_target()

    def train_episode(self):
        del self.memory[:]

    def train_actor(self):
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

        nowVal = self.actorCriticTarget.critic(s)
        nowVal = torch.squeeze(nowVal)

        futureVal = self.actorCriticTarget.critic(s_)
        futureVal = torch.squeeze(futureVal)

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

        futureVal = self.actorCriticTarget.critic(s_)
        futureVal = torch.squeeze(futureVal)

        val = r + self.gamma * futureVal * (1 - done)
        target = val.detach()

        predict = self.actorCriticEval.critic(s)
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

    def get_default_params_path(self, filePath, envName):
        _dirPath = os.path.dirname(os.path.realpath(filePath))
        paramsPath = os.path.join(
            _dirPath, f"params_{envName}_{type(self).__name__}_{self.device.type}.pkl"
        )

        return paramsPath

    def save_models(self, paramsPath):
        print(f"Save parameters to {paramsPath}")
        torch.save(self.actorCriticEval.state_dict(), paramsPath)

    def load_models(self, paramsPath):
        if os.path.exists(paramsPath):
            print(f"Load parameters from {paramsPath}")
            self.actorCriticEval.load_state_dict(
                torch.load(paramsPath, map_location=self.device)
            )
            self.actorCriticTarget.load_state_dict(
                torch.load(paramsPath, map_location=self.device)
            )

    def eval(self):
        print("Evaluation Mode")
        self.actorCriticEval.eval()

    def print_info(self):
        print(f"totalTrainStep:{self.trainStep:7d}")
