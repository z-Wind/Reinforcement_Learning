import torch
import torch.nn.functional as F
from collections import namedtuple
import numpy as np
import os

from Gym.tools.utils import MemoryDataset, OrnsteinUhlenbeckNoise
from .base import IModels

Trajectory = namedtuple(
    "Transition", ("state", "action", "reward", "done", "next_state")
)


class DDPGBase(IModels):
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
        max_noise,
        min_noise,
        #
        learning_rate=0.0001,
        gamma=0.99,
        tau=0.01,
        #
        decayFreq=10 ** 5,
        updateTargetFreq=10000,
        #
        mSize=1_000_000,
        batchSize=32,
        startTrainSize=100,
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
        self.startTrainSize = startTrainSize

        self.lr = learning_rate
        # reward 衰減係數
        self.gamma = gamma
        # target = eval + tau * (target - eval)
        self.tau = tau

        self.updateTargetFreq = updateTargetFreq
        self.decayFreq = decayFreq

        self.noise = OrnsteinUhlenbeckNoise(n_actions)
        self.max_noise = max_noise
        self.min_noise = min_noise

        # optimizer 是訓練的工具
        self.optimizerCritic = optimizerCritic
        self.optimizerActor = optimizerActor

        # 從 1 開始，以免 updateTarget
        self.trainStep = 1

        self.decay_parameters()

    def decay_parameters(self):
        self.noise_amp = max(
            self.min_noise, self.max_noise - self.trainStep / self.decayFreq
        )

    def choose_action(self, state):
        state = torch.from_numpy(state).float()
        state = torch.unsqueeze(state, dim=0).to(self.device)

        if self.actorCriticEval.training:
            # get action based on observation, use exploration policy here
            action = self.get_exploration_action(state)
        else:
            # validate every nth episode
            action = self.get_exploitation_action(state)

        action = action.cpu().data.numpy()
        return action

    def get_exploitation_action(self, state):
        action = self.actorCriticEval.action(state)

        return action

    def get_exploration_action(self, state):
        action = self.actorCriticEval.action(state)
        noise = self.noise() * self.noise_amp

        action = action + torch.from_numpy(noise).float().to(self.device)

        return action

    def store_trajectory(self, s, a, r, done, s_):
        self.memory.add(s, a, r, done, s_)

    def train_step(self):
        if len(self.memory) < self.startTrainSize:
            return

        self.trainStep += 1

        self.train_criticTD()
        self.train_actor()

        self.decay_parameters()
        self.update_target()

    def train_episode(self):
        pass

    def train_actor(self):
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
        # print(list(self.actorCriticEval.actor.parameters()))
        # print("=============================================")

        return loss

    def train_criticTD(self):
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

        _, futureVal = self.actorCriticTarget(s_)
        futureVal = torch.squeeze(futureVal)

        val = r + self.gamma * futureVal * (1 - done)
        target = val.detach()

        predict = self.actorCriticEval.qValue(s, a)
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
            self.noise_amp = self.min_noise
            self.max_noise = self.min_noise

    def eval(self):
        print("Evaluation Mode")
        self.actorCriticEval.eval()

    def print_info(self):
        print(f"totalTrainStep:{self.trainStep:7d} noise_amp:{self.noise_amp:.2f}")

