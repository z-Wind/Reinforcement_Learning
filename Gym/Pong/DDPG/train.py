import gym
import torch

from .DDPG import DDPG
from Gym.tools.utils import env_run
from Gym.tools.atari_wrappers import wrap_env

env = wrap_env(gym.make("PongDeterministic-v4"))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

agent = DDPG(
    device=device,
    n_actions=env.action_space.n,
    n_features=256,
    img_shape=env.observation_space.shape,
    learning_rate=0.001,
    gamma=0.99,
    tau=0.01,
    noiseStart=env.action_space.n,
    noiseEnd=0.2,
    noiseDecayFreq=10000,
    updateTargetFreq=10000,
    mSize=1000000,
    batchSize=100,
    startTrainSize=100,
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
        stopRewardFunc=lambda x: x > 19,
        RENDER=RENDER,
        test=False,
    )
