import gym
from DDPG import DDPG
import matplotlib.pyplot as plt
import torch
import numpy as np
import os

RENDER = False  # 顯示模擬會拖慢運行速度, 等學得差不多了再顯示

env = gym.make("CartPole-v1")
env.seed(1)  # 固定隨機種子 for 再現性
# env = env.unwrapped  # 不限定 episode

print("actions", env.action_space)
# print("actions high", env.action_space.high)
# print("actions low", env.action_space.low)
print("observartions", env.observation_space)
print("observartions high", env.observation_space.high)
print("observartions low", env.observation_space.low)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

agent = DDPG(
    device=device,
    n_actions=1,
    n_actionRange=((env.action_space.n - 1) / 5, 0),
    n_features=env.observation_space.shape[0],
    learning_rate=0.0001,
    gamma=0.999,
    tau=0.01,
    mSize=10000,
    batchSize=100,
)

_dirPath = os.path.dirname(os.path.realpath(__file__))
_dir = os.path.basename(_dirPath)
paramsPath = os.path.join(
    _dirPath, f"params_{env.unwrapped.spec.id}_{_dir}_{device.type}.pkl"
)

if os.path.exists(paramsPath):
    agent.actorCriticEval.load_state_dict(torch.load(paramsPath, map_location=device))
    agent.actorCriticEval.train()

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
    sumR = 0
    for t in range(3000):  # Don't infinite loop while learning
        if RENDER:
            env.render()

        action = agent.choose_action(state, t)
        a = np.clip(np.round(action[0]).astype(int), 0, 1)
        state_, reward, done, _ = env.step(a)
        agent.store_trajectory(state, action, reward, done, state_)

        agent.trainCriticTD()
        agent.trainActor()

        sumR += reward
        if done:
            break

        state = state_

    agent.updateTarget()

    reward_history.append(sumR)
    if RENDER:
        plot_durations()

    avgR = sum(reward_history[:-11:-1]) / 10
    if avgR > maxR:
        maxR = avgR
    print(
        f"episode: {n_episode:4d}\tduration: {t:4d}\tReward: {sumR:5.1f}\tavgR: {avgR:5.1f}\tmaxR: {maxR:5.1f}"
    )

    # 訓練成功條件
    if avgR >= 500 and n_episode > 10:
        break

    # 儲存 model 參數
    torch.save(agent.actorCriticEval.state_dict(), paramsPath)

# 儲存 model 參數
torch.save(agent.actorCriticEval.state_dict(), paramsPath)