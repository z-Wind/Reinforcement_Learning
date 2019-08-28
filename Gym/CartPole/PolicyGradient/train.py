import gym
import torch

from .PolicyGradient import PolicyGradient
from Gym.tools.utils import env_run

env = gym.make("CartPole-v1")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

agent = PolicyGradient(
    device=device,
    n_actions=env.action_space.n,
    n_features=env.observation_space.shape[0],
    learning_rate=0.001,
    gamma=0.99,
    mSize=10000,
)

if __name__ == "__main__":
    RENDER = False  # 顯示模擬會拖慢運行速度, 等學得差不多了再顯示
    env.seed(1)  # 固定隨機種子 for 再現性
    # env = env.unwrapped  # 不限定 episode
    torch.manual_seed(500)  # 固定隨機種子 for 再現性

    env_run(
        env=env,
        agent=agent,
        callerPath=__file__,
        stopRewardFunc=lambda x: x >= 500,
        RENDER=RENDER,
        test=False,
        avgN=20,
    )
