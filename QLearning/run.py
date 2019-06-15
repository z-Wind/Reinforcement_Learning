import gym
from QLearning import QLearning
import matplotlib.pyplot as plt
import torch

RENDER = True  # 顯示模擬會拖慢運行速度, 等學得差不多了再顯示

env = gym.make("CartPole-v0")
env.seed(1)  # 固定隨機種子 for 再現性
# env = env.unwrapped # 不限定 episode

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

agent = QLearning(
    n_features=env.observation_space.shape[0],
    n_actions=env.action_space.n,
    learning_rate=0.001,
)

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


for n_episode in range(3000):
    state = env.reset()
    sumR = 0
    for t in range(3000):  # Don't infinite loop while learning
        if RENDER:
            env.render()

        action = agent.choose_action(state)
        state_, reward, done, _ = env.step(action)
        agent.store_trajectory(state, action, reward, state_)

        sumR += reward

        if done:
            break

        state = state_

        agent.train()

    reward_history.append(sumR)
    if RENDER:
        plot_durations()
    print("episode:", n_episode, "duration:", t, "Reward", sumR)
    
    if n_episode > 2500:
        RENDER = True

