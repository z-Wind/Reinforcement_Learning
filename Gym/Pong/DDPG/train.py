import gym
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import numpy as np
import os
import time

from .DDPG import DDPG
from Gym.tools.atari_wrappers import wrap_env
from Gym.tools.utils import plot_durations


RENDER = False  # 顯示模擬會拖慢運行速度, 等學得差不多了再顯示

env = wrap_env(gym.make("PongDeterministic-v4"))
env.seed(1)  # 固定隨機種子 for 再現性
# env = env.unwrapped  # 不限定 episode

print("actions", env.action_space)
# print("actions high", env.action_space.high)
# print("actions low", env.action_space.low)
print("observartions", env.observation_space)
# print("observartions high", env.observation_space.high)
# print("observartions low", env.observation_space.low)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# data_transform = transforms.Compose(
#     [
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
#     ]
# )
agent = DDPG(
    device=device,
    n_actions=env.action_space.n,
    n_actionRange=(env.action_space.n, 0),
    n_features=256,
    img_shape=env.observation_space.shape,
    learning_rate=0.001,
    gamma=0.99,
    tau=0.01,
    epsilonStart=0,
    epsilonUpdateSteps=10 ** 5,
    updateTargetFreq=10000,
    mSize=1_000_000,
    batchSize=100,
    startTrainSize=100,
    # transforms=data_transform,
)

_dirPath = os.path.dirname(os.path.realpath(__file__))
_dir = os.path.basename(_dirPath)
paramsPath = os.path.join(
    _dirPath, f"params_{env.unwrapped.spec.id}_{_dir}_{device.type}.pkl"
)

if os.path.exists(paramsPath):
    print(f"Load parameters from {paramsPath}")
    agent.actorCriticEval.load_state_dict(torch.load(paramsPath, map_location=device))
    agent.actorCriticTarget.load_state_dict(torch.load(paramsPath, map_location=device))
    agent.actorCriticEval.train()
    agent.epsilonStart = 0.98


reward_history = []


maxR = float("-inf")
for n_episode in range(3000):
    state = env.reset()
    startTime = time.time()
    sumR = 0

    for t in range(10 ** 5):  # Don't infinite loop while learning
        if RENDER:
            env.render()

        action = agent.choose_action(state)
        a = np.argmax(action)
        state_, reward, done, _ = env.step(a)
        agent.store_trajectory(state, action, reward, done, state_)

        agent.train()
        agent.decay_epsilon()
        agent.updateTarget()

        sumR += reward
        if done:
            break

        state = state_

    reward_history.append(sumR)
    if RENDER:
        plot_durations(reward_history)

    avgR = sum(reward_history[:-11:-1]) / 10

    # 訓練成功條件
    if avgR > 19 and n_episode > 10:
        break

    if avgR > maxR and n_episode > 10:
        maxR = avgR
        print("Saving model after {} episodes...".format(n_episode))
        # 儲存 model 參數
        torch.save(agent.actorCriticEval.state_dict(), paramsPath)

    print(
        f"episode: {n_episode:4d} totalTrainStep: {agent.trainStep:7d} duration: {t:4d} Reward: {sumR:5.1f} avgR: {avgR:5.1f} maxR: {maxR:5.1f} epsilon:{agent.epsilon:1.2f} maxAction:{agent.max_noise_action:1.2f} durationTime:{time.time()-startTime:2.2f}"
    )

# 儲存最佳 model 參數
torch.save(agent.actorCriticEval.state_dict(), paramsPath + ".best")
