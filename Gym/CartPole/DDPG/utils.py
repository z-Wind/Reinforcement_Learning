import random
import numpy as np
from torch.utils.data import Dataset
from collections import deque


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


# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.X = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.X = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.X)
        dx = dx + self.sigma * np.random.randn(len(self.X))
        self.X = self.X + dx
        return self.X
