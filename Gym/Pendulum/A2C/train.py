import gym
import torch

from .A2C import A2C
from Gym.tools.utils import env_run

env = gym.make("Pendulum-v0")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

agent = A2C(
    device=device,
    n_actions=env.action_space.shape[0],
    n_features=env.observation_space.shape[0],
    max_actions=env.action_space.high,
    learning_rate=0.0001,
    gamma=1,
    tau=0.001,
    updateTargetFreq=500,
    mSize=1000,
    batchSize=10,
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
        stopRewardFunc=lambda x: x > -100,
        RENDER=RENDER,
        test=False,
    )
