import abc
import os

from Gym.tools.utils import MemoryDataset


class IModels(abc.ABC):
    @abc.abstractmethod
    def __init__(self, device, mSize, transforms):
        self.device = device
        self.memory = MemoryDataset(mSize, transforms)

    @abc.abstractmethod
    def choose_action(self, state):
        pass

    @abc.abstractmethod
    def train_step(self):
        pass

    @abc.abstractmethod
    def train_episode(self):
        pass

    @abc.abstractmethod
    def save_models(self, path):
        pass

    @abc.abstractmethod
    def load_models(self, path):
        pass

    @abc.abstractmethod
    def eval(self):
        pass

    @abc.abstractmethod
    def print_info(self):
        pass

    def store_trajectory(self, state, action, reward, done, next_state):
        self.memory.add(state, action, reward, done, next_state)

    def get_default_params_path(self, filePath, envName):
        _dirPath = os.path.dirname(os.path.realpath(filePath))
        paramsPath = os.path.join(
            _dirPath, f"params_{envName}_{type(self).__name__}_{self.device.type}.pkl"
        )

        return paramsPath

    def get_default_png_path(self, filePath, envName):
        _dirPath = os.path.dirname(os.path.realpath(filePath))
        pngPath = os.path.join(
            _dirPath, f"train_{envName}_{type(self).__name__}_{self.device.type}.png"
        )

        return pngPath
