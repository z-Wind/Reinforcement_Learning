import random
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import deque
import cv2
from matplotlib import pyplot as plt
import time


class MemoryDataset(Dataset):
    def __init__(self, size, transforms=None):
        self.memory = deque(maxlen=size)
        self.transforms = transforms

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, idx):
        sample = self.memory[idx]

        if self.transforms:
            sample = self.transforms(sample)

        return sample

    def add(self, s, a, r, done, s_):
        """
        adds a particular transaction in the memory buffer
        :param s: current state
        :param a: action taken
        :param r: reward received
        :param done: env finish
        :param s_: next state
        :return:
        """
        self.memory.append([s, a, r, done, s_])

    def sample(self, batchSize):
        """
        samples a random batch from the replay memory buffer
        :param batchSize: batch size
        :return: batch (numpy array)
        """
        batchSize = min(batchSize, self.__len__())
        batch = random.sample(self.memory, batchSize)

        return batch


class OrnsteinUhlenbeckNoise:
    """
    A Ornstein Uhlenbeck action noise, this is designed to aproximate brownian motion with friction.

    Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab

    :param dim: (tuple) the dimension of the noise
    :param mu: (float) the mean of the noise
    :param theta: (float) the rate of mean reversion, affect converge
    :param sigma: (float) the scale of the noise, affect random
    :param dt: (float) the timestep for the noise
    """

    def __init__(self, dim, mu=0, theta=0.15, sigma=0.2, dt=1.0):
        self.dim = dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt

        self.X = np.ones(self.dim) * self.mu

    def reset(self):
        self.X = np.ones(self.dim) * self.mu

    def __call__(self):
        drift = self.theta * (self.mu - self.X) * self.dt
        random = self.sigma * self._delta_wiener()

        self.X = self.X + drift + random

        return self.X

    def _delta_wiener(self):
        return np.sqrt(self.dt) * np.random.randn(self.dim)


def imshow(img):
    plt.imshow(img)
    plt.show()


def plot_durations(reward_history):
    y_t = torch.FloatTensor(reward_history)
    plt.figure(1)
    plt.clf()
    plt.title("Training...")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.plot(y_t.numpy())
    # Take 100 episode averages and plot them too
    if len(reward_history) >= 100:
        means = y_t.unfold(dim=0, size=100, step=1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated


def env_run(
    env,
    agent,
    callerPath,
    stopRewardFunc=None,
    RENDER=False,
    test=False,
    showEnvInfo=True,
):
    if showEnvInfo:
        if hasattr(env, "action_space"):
            print("action_space", env.action_space)

            if hasattr(env.action_space, "n"):
                print("discrete action")
                print("action_space n", env.action_space.n)
            if hasattr(env.action_space, "high"):
                print("continuos action")
                print("action_space high", env.action_space.high)
                print("action_space low", env.action_space.low)

        if hasattr(env, "observation_space"):
            print("observation_space", env.observation_space)

            if hasattr(env.observation_space, "high"):
                print("observation_space high", env.observation_space.high)
                print("observation_space low", env.observation_space.low)

    if test:
        env_test(env=env, agent=agent, callerPath=callerPath, RENDER=RENDER)

    if stopRewardFunc is None:
        print("Should set stopReward Function to stop training")
        return

    env_train(
        env=env,
        agent=agent,
        callerPath=callerPath,
        stopRewardFunc=stopRewardFunc,
        RENDER=RENDER,
    )


def env_train(env, agent, stopRewardFunc, callerPath, RENDER=False):
    paramsPath = agent.get_default_params_path(callerPath, env.unwrapped.spec.id)
    agent.load_models(paramsPath)

    print("paramsPath:", paramsPath)

    reward_history = []
    maxR = float("-inf")
    for n_episode in range(3000):
        state = env.reset()
        startTime = time.time()
        sumR = 0

        for n_step in range(10 ** 5):  # Don't infinite loop while learning
            if RENDER:
                env.render()

            action, a = agent.choose_action(state)
            state_, reward, done, _ = env.step(a)

            agent.store_trajectory(state, action, reward, done, state_)
            agent.train_step()

            sumR += reward
            if done:
                break

            state = state_

        agent.train_episode()

        reward_history.append(sumR)
        if RENDER:
            plot_durations(reward_history)

        avgR = sum(reward_history[:-11:-1]) / 10

        # 訓練成功條件
        if stopRewardFunc(avgR) and n_episode > 10:
            break

        if avgR > maxR and n_episode > 10:
            maxR = avgR
            # 儲存 model 參數
            agent.save_models(paramsPath)

        print(
            f"episode:{n_episode:4d} duration:{n_step:4d} Reward:{sumR:5.1f} avgR:{avgR:5.1f} maxR:{maxR:5.1f} durationTime:{time.time()-startTime:2.2f}",
            end=" ",
        )
        agent.print_info()

    # 儲存最佳 model 參數
    agent.save_models(paramsPath + ".best")


def env_test(env, agent, callerPath, RENDER=False):
    paramsPath = agent.get_default_params_path(callerPath, env.unwrapped.spec.id)
    paramsPath += ".best"
    print("paramsPath:", paramsPath)
    agent.load_models(paramsPath)
    agent.eval()

    reward_history = []
    maxR = float("-inf")
    for n_episode in range(100):
        state = env.reset()
        sumR = 0
        for n_step in range(10 ** 5):  # Don't infinite loop while learning
            if RENDER:
                env.render()

            action, a = agent.choose_action(state)
            state_, reward, done, _ = env.step(a)

            sumR += reward
            if done:
                break

            state = state_

        reward_history.append(sumR)
        if RENDER:
            plot_durations(reward_history)

        avgR = sum(reward_history[:-11:-1]) / 10

        if avgR > maxR and n_episode > 10:
            maxR = avgR

        print(
            f"episode:{n_episode:4d} duration:{n_step:4d} Reward:{sumR:5.1f} avgR: {avgR:5.1f}  maxR:{maxR:5.1f}"
        )
