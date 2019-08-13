import gym
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import os
import numpy as np
import time

from .QLearning import QLearning
from .atari_wrappers import wrap_env
from .utils import imshow

RENDER = False  # 顯示模擬會拖慢運行速度, 等學得差不多了再顯示

env = wrap_env(gym.make("PongDeterministic-v4"))
env.seed(1)  # 固定隨機種子 for 再現性
# env = env.unwrapped # 不限定 episode

print("action", env.action_space)
print("observation", env.observation_space)
# print(env.observation_space.high)
# print(env.observation_space.low)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# data_transform = transforms.Compose(
#    [
#        transforms.ToTensor(),  # to [0, 1]
#        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # to [-1, 1]
#    ]
# )

agent = QLearning(
    device=device,
    n_actions=env.action_space.n,
    img_shape=env.observation_space.shape,
    learning_rate=0.0001,
    gamma=0.99,
    tau=0.01,
    epsilonStart=0.0,
    epsilonUpdateSteps=10 ** 5,
    updateTargetFreq=10000,
    mSize=1_000_000,
    batchSize=32,
    startTrainLen=100,
    # transforms=data_transform,
)

_dirPath = os.path.dirname(os.path.realpath(__file__))
_dir = os.path.basename(_dirPath)
paramsPath = os.path.join(
    _dirPath, f"params_{env.unwrapped.spec.id}_{_dir}_{device.type}.pkl"
)

if os.path.exists(paramsPath):
    agent.net.load_state_dict(torch.load(paramsPath, map_location=device))
    agent.net.train()

reward_history = []


def plot_durations():
    y_t = torch.FloatTensor(reward_history)
    plt.figure(1)
    plt.clf()
    plt.title("Training...")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.plot(y_t.numpy())
    # Take 100 episode averages and plot them too
    if len(reward_history) >= 100:
        means = y_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated


maxR = float("-inf")
for n_episode in range(3000):
    state = env.reset()
    # imshow(state[0, :, :])
    sumR = 0
    startTime = time.time()
    t = 0

    for t in range(3000):  # Don't infinite loop while learning
        if RENDER:
            env.render()

        action = agent.choose_action(state)
        state_, reward, done, _ = env.step(action)
        agent.store_trajectory(state, action, reward, done, state_)

        sumR += reward

        agent.train()

        if done:
            break

        state = state_

        agent.updateTarget()
        agent.decay_epsilon()

    reward_history.append(sumR)
    if RENDER:
        plot_durations()

    avgR = sum(reward_history[:-11:-1]) / 10

    # 訓練成功條件
    if avgR > 19 and n_episode > 10:
        break

    if avgR > maxR and n_episode > 10:
        maxR = avgR
        print("Saving model after {} episodes...".format(n_episode))
        # 儲存 model 參數
        torch.save(agent.net.state_dict(), paramsPath)

    print(
        f"episode: {n_episode:4d} totalTrainStep: {agent.trainStep:7d} duration: {t:4d} Reward: {sumR:5.1f} avgR: {avgR:5.1f} maxR: {maxR:5.1f} epsilon:{agent.epsilon:1.2f} durationTime:{time.time()-startTime:2.2f}"
    )

# 儲存最佳 model 參數
torch.save(agent.net.state_dict(), paramsPath + ".best")
