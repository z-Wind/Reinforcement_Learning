import gym
import matplotlib.pyplot as plt
import torch
import os

from .DDPG import DDPG

RENDER = False  # 顯示模擬會拖慢運行速度, 等學得差不多了再顯示

env = gym.make("Pendulum-v0")
env.seed(1)  # 固定隨機種子 for 再現性
# env = env.unwrapped  # 不限定 episode

print("actions", env.action_space)
print("actions high", env.action_space.high)
print("actions low", env.action_space.low)
print("observartions", env.observation_space)
print("observartions high", env.observation_space.high)
print("observartions low", env.observation_space.low)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

agent = DDPG(
    device=device,
    n_actions=env.action_space.shape[0],
    n_actionRange=zip(env.action_space.high, env.action_space.low),
    n_features=env.observation_space.shape[0],
    learning_rate=0.001,
    gamma=0.99,
    tau=0.001,
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
    agent.actorCriticTarget.load_state_dict(torch.load(paramsPath, map_location=device))
    agent.actorCriticEval.train()

reward_history = []




for n_episode in range(3000):
    state = env.reset()
    sumR = 0
    for t in range(3000):  # Don't infinite loop while learning
        if RENDER:
            env.render()

        action = agent.choose_action(state, t)
        state_, reward, done, _ = env.step(action)

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
        plot_durations(reward_history)

    avgR = sum(reward_history[:-11:-1]) / 10
    print(
        "episode: {:4d} duration: {:4d} Reward: {:5.1f} avgR: {:5.1f}".format(
            n_episode, t, sumR, avgR
        )
    )

    # 訓練成功條件
    if avgR > -100 and n_episode > 10:
        break

    # 儲存 model 參數
    torch.save(agent.actorCriticEval.state_dict(), paramsPath)

# 儲存最佳 model 參數
torch.save(agent.actorCriticEval.state_dict(), paramsPath + ".best")
