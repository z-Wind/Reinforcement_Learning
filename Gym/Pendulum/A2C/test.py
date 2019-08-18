import gym
import matplotlib.pyplot as plt
import torch
import os

from .A2C import A2C

RENDER = True  # 顯示模擬會拖慢運行速度, 等學得差不多了再顯示

env = gym.make("Pendulum-v0")

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

agent = A2C(
    device=device,
    n_actions=env.action_space.shape[0],
    n_actionRange=zip(env.action_space.high, env.action_space.low),
    n_features=env.observation_space.shape[0],
    learning_rate=0.01,
    gamma=0.9,
)

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
        state_, reward, done, _ = env.step(action)

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
