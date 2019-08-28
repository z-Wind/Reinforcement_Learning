from Gym.tools.utils import env_run
from .train import env, agent

if __name__ == "__main__":
    RENDER = True  # 顯示模擬會拖慢運行速度, 等學得差不多了再顯示

    env_run(env=env, agent=agent, callerPath=__file__, RENDER=RENDER, test=True)
