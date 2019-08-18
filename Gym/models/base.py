import abc


class IModels(abc.ABC):
    @abc.abstractmethod
    def __init__(self, device):
        pass

    @abc.abstractmethod
    def choose_action(self, state):
        pass

    @abc.abstractmethod
    def store_trajectory(self, state, action, reward, done, next_state):
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

