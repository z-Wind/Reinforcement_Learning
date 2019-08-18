import gym

import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import os
import numpy as np

from .DDPG import DDPG
from Gym.tools.atari_wrappers import wrap_env

RENDER = True  # 顯示模擬會拖慢運行速度, 等學得差不多了再顯示

env = wrap_env(gym.make("PongDeterministic-v4"))

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[128, 128, 128], std=[128, 128, 128]),
    ]
)

agent = DDPG(
    device=device,
    n_actions=env.action_space.n,
    n_actionRange=(1, 0),
    n_features=256,
    img_shape=env.observation_space.shape,
    learning_rate=0.001,
    gamma=0.99,
    tau=0.01,
    mSize=1_000_000,
    batchSize=100,
    # transforms=data_transform,
)

_dir = os.path.dirname(os.path.realpath(__file__))
_dirPath = os.path.dirname(os.path.realpath(__file__))
_dir = os.path.basename(_dirPath)
paramsPath = os.path.join(
    _dirPath, f"params_{env.unwrapped.spec.id}_{_dir}_{device.type}.pkl.best"
)

agent.actorCriticEval.load_state_dict(torch.load(paramsPath, map_location=device))
agent.actorCriticEval.eval()

reward_history = []




for n_episode in range(3000):
    state = env.reset()
    sumR = 0
    for t in range(3000):  # Don't infinite loop while learning
        if RENDER:
            env.render()

        action = agent.choose_action(state)
        a = np.argmax(action)
        print(a)
        state_, reward, done, _ = env.step(a)

        sumR += reward
        if done:
            break

        state = state_

    reward_history.append(sumR)
    if RENDER:
        plot_durations(reward_history)

    avgR = sum(reward_history[:-11:-1]) / 10
    print(
        "episode: {:4d} duration: {:4d} Reward: {:5.1f} avgR: {:5.1f}".format(
            n_episode, t, sumR, avgR
        )
    )
