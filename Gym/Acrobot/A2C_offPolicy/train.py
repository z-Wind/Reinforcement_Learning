import gym
import torch

from .A2C import A2C
from Gym.tools.utils import env_run

env = gym.make("Acrobot-v1")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

agent = A2C(
    device=device,
    n_actions=env.action_space.n,
    n_features=env.observation_space.shape[0],
    learning_rate=0.001,
    gamma=0.9,
    tau=0.01,
    updateTargetFreq=1000,
    mSize=10000,
    batchSize=100,
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
        stopRewardFunc=lambda x: x >= -80,
        RENDER=RENDER,
        test=False,
    )
