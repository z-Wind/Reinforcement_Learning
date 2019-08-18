import gym
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import numpy as np
import os
import time

from .DDPG.DDPG import DDPG
from .QLearning.QLearning import QLearning
from Gym.tools.atari_wrappers import wrap_env


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

_dirPath = os.path.dirname(os.path.realpath(__file__))
paramsPath = os.path.join(
    _dirPath,
    "QLearning",
    f"params_{env.unwrapped.spec.id}_QLearning_{device.type}.pkl.best",
)
agentTeacher.net.load_state_dict(torch.load(paramsPath, map_location=device))
agentTeacher.net.eval()

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

_dir = os.path.basename(_dirPath)
paramsPath = os.path.join(
    _dirPath, f"params_{env.unwrapped.spec.id}_{_dir}_{device.type}.pkl"
)

if os.path.exists(paramsPath):
    agentStudent.actorCriticEval.load_state_dict(
        torch.load(paramsPath, map_location=device)
    )
    agentStudent.actorCriticEval.train()


reward_history = []
print("paramsPath", paramsPath)




maxR = float("-inf")
for n_episode in range(3000):
    state = env.reset()
    startTime = time.time()
    sumR = 0

    # teach
    agentStudent.actorCriticEval.train()
    for t in range(3000):  # Don't infinite loop while learning
        if RENDER:
            env.render()

        action = agentTeacher.choose_action(state)
        actionStudent = agentStudent.choose_action(state)
        # print(actionStudent[0])
        actionStudent = np.argmax(actionStudent)
        state_, reward, done, _ = env.step(action)

        action_onehot = np.zeros(env.action_space.n)
        action_onehot[action] = 1
        # print(action_onehot)
        agentStudent.store_trajectory(state, action_onehot, reward, done, state_)

        loss = agentStudent.teach()
        # print(action, actionStudent)
        # print("==================================")

        if done:
            break

        state = state_

    agentStudent.updateTarget()

    # Test
    state = env.reset()
    agentStudent.actorCriticEval.eval()
    for t in range(3000):  # Don't infinite loop while learning
        if RENDER:
            env.render()

        action = agentStudent.choose_action(state)
        state_, reward, done, _ = env.step(np.argmax(action))

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
        torch.save(agentStudent.actorCriticEval.state_dict(), paramsPath)

    print(
        f"episode: {n_episode:4d} totalTrainStep: {agentStudent.trainStep:7d} duration: {t:4d} Reward: {sumR:5.1f} avgR: {avgR:5.1f} maxR: {maxR:5.1f} epsilon:{agentStudent.epsilon:1.2f} maxAction:{agentStudent.max_noise_action:1.2f} durationTime:{time.time()-startTime:2.2f}"
    )

# 儲存最佳 model 參數
torch.save(agentStudent.actorCriticEval.state_dict(), paramsPath + ".best")
