from .QLearning import QLearning
from .atrain import PongAgent
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

agent1 = PongAgent(gamma=0.99, size=1_000_000)
agent2 = QLearning(
    device=device,
    n_actions=env.action_space.n,
    img_shape=env.observation_space.shape,
    learning_rate=0.001,
    gamma=0.99,
    tau=0.01,
    mSize=10000,
    batchSize=100,
)

paramsPath = ""
agent1.dqn.load_state_dict(torch.load(paramsPath, map_location=device))
agent1.target_dqn.load_state_dict(torch.load(paramsPath, map_location=device))
agent2.net.load_state_dict(torch.load(paramsPath, map_location=device))
agent2.netTarget.load_state_dict(torch.load(paramsPath, map_location=device))


state = np.ones((4,84,84))
action1 = agent1.choose_action(state)
action2 = agent2.choose_action(state)

agent1.train(
    replay_buffer_fill_len=100,
    batch_size=32,
    episodes=10 ** 5,
    stop_reward=19,
    sync_target_net_freq=10000,
)
agent2.train()
