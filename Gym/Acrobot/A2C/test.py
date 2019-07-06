import gym
from A2C import A2C
import matplotlib.pyplot as plt
import torch

RENDER = True  # 顯示模擬會拖慢運行速度, 等學得差不多了再顯示

env = gym.make("Acrobot-v1")

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

agent = A2C(
    n_actions=env.action_space.n,
    n_features=env.observation_space.shape[0],
    learning_rate=0.01,
    gamma=0.9,
)
agent.actorCriticEval.load_state_dict(torch.load("params.pkl"))
agent.actorCriticEval.eval()

reward_history = []


def plot_durations():
    y_t = torch.FloatTensor(reward_history)
    plt.figure(1)
    plt.clf()
    plt.title("Testing...")
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

        sumR += reward
        if done:
            break

        state = state_

    reward_history.append(sumR)
    if RENDER:
        plot_durations()

    avgR = sum(reward_history[:-11:-1]) / 10
    print(
        "episode: {:4d} duration: {:4d} Reward: {:5.1f} avgR: {:5.1f}".format(
            n_episode, t, sumR, avgR
        )
    )
