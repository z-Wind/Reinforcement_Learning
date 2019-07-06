import torch
import random
from torch.utils.data import Dataset
from collections import deque


class MemoryDataset(Dataset):
    def __init__(self, size, transform=None):
        self.memory = deque(maxlen=size)
        self.transform = transform

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, idx):
        sample = self.memory[idx]

        if self.transform:
            sample = self.transform(sample)

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
