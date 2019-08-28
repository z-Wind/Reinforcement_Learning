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
        #
        learning_rate,
        gamma,
        tau,
        #
        updateTargetFreq,
        #
        noiseStart,
        noiseEnd,
        noiseDecayFreq,
        #
        mSize,
        batchSize,
        startTrainSize,
        #
        transforms,
    ):
        """
        gamma: reward 的衰減係數
        tau: update target 的係數，target = eval + tau * (target - eval)
        noiseStart: noise 的起始值，變化的幅度，越大越隨機
        noiseEnd: noise 的終值
        noiseDecayFreq: noise 衰減頻率
        updateTargetFreq: 更新 target 的頻率
        startTrainSize: 當 memory size 大於此值時，便開始訓練
        """
        super().__init__(device, mSize, transforms)

        self.device = device

        self.actorCriticEval = actorCriticEval.to(self.device)
        self.actorCriticTarget = actorCriticTarget.to(self.device)

        print("device", self.device)
        print(self.actorCriticEval)

        # self.memory = MemoryDataset(mSize, transforms)
        self.batchSize = batchSize
        self.startTrainSize = startTrainSize

        self.lr = learning_rate
        # reward 衰減係數
        self.gamma = gamma
        # target = eval + tau * (target - eval)
        self.tau = tau

        self.updateTargetFreq = updateTargetFreq
        self.noiseDecayFreq = noiseDecayFreq

        self.noise = OrnsteinUhlenbeckNoise(n_actions)
        self.noiseStart = noiseStart
        self.noiseAmp = noiseStart
        self.noiseEnd = noiseEnd

        # optimizer 是訓練的工具
        self.optimizerCritic = optimizerCritic
        self.optimizerActor = optimizerActor

        # 從 1 開始，以免 updateTarget
        self.trainStep = 1

        self.decay_parameters()

    def decay_parameters(self):
        # 因 self.trainStep = 1 開始，需減 1
        self.noiseAmp = max(
            self.noiseEnd, self.noiseStart - self.trainStep / self.noiseDecayFreq
        )

    def choose_action(self, state):
        state = torch.from_numpy(state).float()
        state = torch.unsqueeze(state, dim=0).to(self.device)

        if self.actorCriticEval.training:
            # 加入隨機 noise
            action = self.get_exploration_action(state)
        else:
            action = self.get_exploitation_action(state)

        action = action.cpu().data.numpy()
        return action

    def get_exploitation_action(self, state):
        action = self.actorCriticEval.action(state)

        return action

    def get_exploration_action(self, state):
        action = self.actorCriticEval.action(state)
        noise = self.noise() * self.noiseAmp

        action = action + torch.from_numpy(noise).float().to(self.device)

        return action

    def store_trajectory(self, s, a, r, done, s_):
        self.memory.add(s, a, r, done, s_)

    def train_step(self):
        if len(self.memory) < self.startTrainSize:
            return

        self.train_criticTD()
        self.train_actor()
        self.decay_parameters()

        self.trainStep += 1

        self.update_target()

    def train_episode(self):
        pass

    def train_actor(self):
        batch = Trajectory(*zip(*self.memory.sample(self.batchSize)))

        # 轉成 np.array 加快轉換速度
        s = np.array(batch.state)
        s = torch.FloatTensor(s).to(self.device)

        a = self.actorCriticEval.action(s)
        qVal = self.actorCriticEval.criticism(s, a)
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

        predict = self.actorCriticEval.criticism(s, a)
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
        self.noiseAmp = self.noiseEnd
        self.noiseStart = self.noiseEnd

        return True

    def eval(self):
        print("Evaluation Mode")
        self.actorCriticEval.eval()

    def print_info(self):
        print(f"totalTrainStep:{self.trainStep:7d} noiseAmp:{self.noiseAmp:.2f}")
