import gym
import matplotlib.pyplot as plt
import torch
import os

from .A2C import A2C

RENDER = False  # 顯示模擬會拖慢運行速度, 等學得差不多了再顯示

env = gym.make("CartPole-v1")
env.seed(1)  # 固定隨機種子 for 再現性
# env = env.unwrapped  # 不限定 episode

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

agent = A2C(
    device=device,
    n_actions=env.action_space.n,
    n_features=env.observation_space.shape[0],
    learning_rate=0.01,
    gamma=0.9,
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

        action = agent.choose_action(state)
        state_, reward, done, _ = env.step(action)
        agent.store_trajectory(state, action, reward, state_)

        agent.trainCriticTD()

        sumR += reward
        if done:
            break

        state = state_

    agent.trainCriticMC()
    agent.trainActor()
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
    if avgR >= 500 and n_episode > 10:
        break

    # 儲存 model 參數
    torch.save(agent.actorCriticEval.state_dict(), paramsPath)

# 儲存最佳 model 參數
torch.save(agent.actorCriticEval.state_dict(), paramsPath + ".best")
