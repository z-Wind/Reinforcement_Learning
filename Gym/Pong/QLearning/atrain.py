import time

import gym
import numpy as np
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import os

import datetime
from .atari_wrappers import wrap_env
from .QLearning import Net, QLearning
from .utils import MemoryDataset

torch.manual_seed(500)  # 固定隨機種子 for 再現性

Trajectory = namedtuple(
    "Transition", ("state", "action", "reward", "done", "next_state")
)


class PongAgent:
    """
    Pong agent. Implements training and testing methods
    """

    def __init__(self, gamma, size, num_actions, batch_size, device, epsilon):
        self.device = device
        self.num_actions = num_actions
        self.dqn = Net((84, 84, 4), self.num_actions).to(self.device)
        self.target_dqn = Net((84, 84, 4), self.num_actions).to(self.device)
        self.batch_size = batch_size

        print(self.dqn)

        self.buffer = MemoryDataset(size)

        self.gamma = gamma
        self.epsilon = epsilon
        self.tau = 0.00

        self.mse_loss = nn.MSELoss()
        self.optim = optim.RMSprop(self.dqn.parameters(), lr=0.0001)

        self.out_dir = "./model"

        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

    def choose_action(self, state):
        choice = np.random.choice([0, 1], p=((1 - self.epsilon), self.epsilon))

        # epslion greedy
        if choice == 0 and self.dqn.training:
            action = np.random.choice(range(self.num_actions), 1)
        else:
            # state = torch.unsqueeze(self.transforms(state), dim=0).to(self.device)
            state = torch.FloatTensor(state)
            state = torch.unsqueeze(state, dim=0).to(self.device)
            qValues = self.dqn(state)
            action_max_value, action = torch.max(qValues, 1)

        return action.item()

    def store_trajectory(self, s, a, r, done, s_):
        self.buffer.add(s, a, r, done, s_)

    def train(self):

        # sample random minibatch of transactions
        batch = Trajectory(*zip(*self.buffer.sample(self.batch_size)))

        s = np.array(batch.state)
        a = np.array(batch.action)
        r = np.array(batch.reward)
        done = np.array(batch.done)
        s_ = np.array(batch.next_state)

        s = torch.FloatTensor(s).to(self.device)
        a = torch.LongTensor(a)
        a = torch.unsqueeze(a, 1).to(self.device)
        r = torch.FloatTensor(r).to(self.device)
        done = torch.FloatTensor(done).to(self.device)
        s_ = torch.FloatTensor(s_).to(self.device)

        # 在 dim=1，以 a 為 index 取值
        qValue = self.dqn(s).gather(1, a).squeeze(1)

        # caculate Target
        qNext = self.target_dqn(s_).detach()  # detach from graph, don't backpropagate
        target = r + self.gamma * qNext.max(1)[0] * (1 - done)

        self.optim.zero_grad()
        loss = self.mse_loss(qValue, target.detach())
        loss.backward()
        self.optim.step()

        # print(list(self.dqn.parameters()))

    # 逐步更新 target NN
    def updateTarget(self):
        print(f"Update target network... tau={self.tau}")
        for paramEval, paramTarget in zip(
            self.dqn.parameters(), self.target_dqn.parameters()
        ):
            paramTarget.data = paramEval.data + self.tau * (
                paramTarget.data - paramEval.data
            )


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    env = wrap_env(gym.make("PongDeterministic-v4"))
    env.seed(1)  # 固定隨機種子 for 再現性

    agent = PongAgent(
        gamma=0.99,
        size=1_000_000,
        num_actions=env.action_space.n,
        batch_size=32,
        device=device,
        epsilon=0.8,
    )
    # agent = QLearning(
    #    device=device,
    #    n_actions=env.action_space.n,
    #    img_shape=env.observation_space.shape,
    #    learning_rate=0.0001,
    #    gamma=0.99,
    #    tau=0.00,
    #    epsilonStart=0.8,
    #    mSize=1_000_000,
    #    batchSize=32,
    #    # transforms=data_transform,
    # )

    total_steps = 0
    running_episode_reward = 0

    # populate replay memory
    print("Populating replay buffer... ")
    print("\n")
    state = env.reset()

    episodes = 10 ** 5
    stop_reward = 19
    sync_target_net_freq = 10000
    replay_buffer_fill_len = 100
    for i in range(replay_buffer_fill_len):
        # action = agent.select_action(state, 1)  # force to choose a random action
        action = agent.choose_action(state)  # force to choose a random action
        state_, reward, done, _ = env.step(action)

        agent.store_trajectory(state, action, reward, done, state_)

        state = state_
        if done:
            env.reset()

    # main loop - iterate over episodes
    for i in range(1, episodes + 1):
        # reset the environment
        done = False
        state = env.reset()

        # reset spisode reward and length
        episode_reward = 0
        episode_length = 0

        # play until it is possible
        while not done:
            # synchronize target network with estimation network in required frequence

            action = agent.choose_action(state)
            # execute action in the environment
            state_, reward, done, _ = env.step(action)
            agent.store_trajectory(state, action, reward, done, state_)
            agent.train()

            # set the state for the next action selction and update counters and reward
            state = state_
            total_steps += 1
            episode_length += 1
            episode_reward += reward

            if (total_steps % sync_target_net_freq) == 0:
                print("synchronizing target network...")
                print("\n")
                agent.updateTarget()

        running_episode_reward = running_episode_reward * 0.9 + 0.1 * episode_reward

        # if (i % 10) == 0 or (running_episode_reward > stop_reward):
        print("global step: {}".format(total_steps))
        print("epsilon: {}".format(agent.epsilon))
        print("episode: {}".format(i))
        print("running reward: {}".format(round(running_episode_reward, 2)))
        print("episode_length: {}".format(episode_length))
        print("episode reward: {}".format(episode_reward))
        print("\n")

        if running_episode_reward > stop_reward:
            break

    print("Finish training at: ")
