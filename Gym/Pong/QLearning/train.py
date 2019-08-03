import gym
from QLearning import QLearning
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import os

RENDER = False  # 顯示模擬會拖慢運行速度, 等學得差不多了再顯示

env = gym.make("Pong-v0")
env.seed(1)  # 固定隨機種子 for 再現性
# env = env.unwrapped # 不限定 episode

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[128, 128, 128], std=[128, 128, 128]),
    ]
)

agent = QLearning(
    device=device,
    n_actions=env.action_space.n,
    img_shape=env.observation_space.shape,
    learning_rate=0.001,
    gamma=0.99,
    tau=0.01,
    mSize=10000,
    batchSize=100,
    transforms=data_transform,
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


for n_episode in range(3000):
    state = env.reset()
    sumR = 0
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

    reward_history.append(sumR)
    if RENDER:
        plot_durations()

    avgR = sum(reward_history[:-11:-1]) / 10
    print(
        "episode: {:4d} duration: {:4d} Reward: {:5.1f} avgR: {:5.1f}".format(
            n_episode, t, sumR, avgR
        )
    )

    # 訓練成功條件
    if avgR > 20 and n_episode > 10:
        break

    # 儲存 model 參數
    torch.save(agent.net.state_dict(), paramsPath)

# 儲存 model 參數
torch.save(agent.net.state_dict(), paramsPath)
