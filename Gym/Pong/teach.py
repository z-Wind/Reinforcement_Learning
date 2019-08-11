import gym
from .DDPG import DDPG
from QLearning.QLearning import QLearning
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import numpy as np
import os
from .atari_wrappers import wrap_env


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
agentStudent = DDPG(
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
_dirPath = os.path.dirname(os.path.realpath(__file__))
_dir = os.path.basename(_dirPath)
paramsPath = os.path.join(
    _dirPath, f"params_{env.unwrapped.spec.id}_{_dir}_{device.type}.pkl"
)

if os.path.exists(paramsPath):
    agentStudent.actorCriticEval.load_state_dict(
        torch.load(paramsPath, map_location=device)
    )
    agentStudent.actorCriticEval.train()

agentTeacher = QLearning(
    device=device,
    n_actions=env.action_space.n,
    img_shape=env.observation_space.shape,
    learning_rate=0.001,
    gamma=0.99,
    tau=0.01,
    mSize=1_000_000,
    batchSize=100,
    # transforms=data_transform,
)
paramsPath = os.path.join(
    _dirPath,
    "QLearning",
    f"params_{env.unwrapped.spec.id}_QLearning_{device.type}.pkl.best",
)

agentTeacher.net.load_state_dict(torch.load(paramsPath, map_location=device))
agentTeacher.net.eval()

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

        action = agentTeacher.choose_action(state)
        state_, reward, done, _ = env.step(action)

        action_onehot = np.zeros(env.action_space.n)
        action_onehot[action] = 1
        agentStudent.store_trajectory(state, action_onehot, reward, done, state_)

        loss = agentStudent.teach()
        print(loss)

        sumR += reward
        if done:
            break

        state = state_

    agentStudent.updateTarget()

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
    if avgR > 20 and n_episode > 10:
        break

    # 儲存 model 參數
    torch.save(agentStudent.actorCriticEval.state_dict(), paramsPath)

# 儲存最佳 model 參數
torch.save(agentStudent.actorCriticEval.state_dict(), paramsPath + ".best")
