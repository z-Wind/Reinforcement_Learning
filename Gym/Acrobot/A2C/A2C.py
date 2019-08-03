import torch
import torch.nn.functional as F
from torch.distributions import Categorical

torch.manual_seed(500)  # 固定隨機種子 for 再現性


class A2C:
    def __init__(self, device, n_actions, n_features, learning_rate=0.01, gamma=0.9):
        self.device = device
        self.actorCriticEval = ActorCriticNet(n_actions, n_features).to(self.device)
        self.actorCriticTarget = ActorCriticNet(n_actions, n_features).to(self.device)
        print(self.device)
        print(self.actorCriticEval)
        print(self.actorCriticTarget)

        self.lr = learning_rate
        # reward 衰減係數
        self.gamma = gamma

        # optimizer 是訓練的工具
        # 傳入 net 的所有參數, 學習率
        self.optimizerActorCriticEval = torch.optim.Adam(
            self.actorCriticEval.parameters(), lr=self.lr
        )

        self.saved_log_probs = []
        self.rewards = []
        self.states = []

    def choose_action(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        probs, _ = self.actorCriticEval(state)

        m = Categorical(probs)
        action = m.sample()

        log_prob = m.log_prob(action)
        self.saved_log_probs.append(log_prob)

        return action.item()

    def store_trajectory(self, s, a, r, s_):
        self.rewards.append(r)
        self.states.append(s)
        self.nextState = s_

    # episode train
    def trainActor(self):
        R = 0
        policy_loss = []
        rewards = []

        # 現在的 reward 是由現在的 action 造成的，過去的 action 影響不大
        # 越未來的 reward，現在的 action 影響會越來越小
        # 若是看直接看 total reward 不太能區分出 action 的好壞，導致學習不好
        nextStates = self.states + [self.nextState]
        for r, s, s_ in zip(self.rewards[::-1], self.states[::-1], nextStates[::-1]):
            _, futureVal = self.actorCriticTarget(
                torch.tensor(s_).float().to(self.device)
            )
            _, nowVal = self.actorCriticTarget(torch.tensor(s).float().to(self.device))
            R_now = r + futureVal.detach() - nowVal.detach()
            R = R_now + self.gamma * R
            rewards.insert(0, R)

        for log_prob, reward in zip(self.saved_log_probs, rewards):
            # 最大化，故加上負號
            policy_loss.append(-log_prob * reward)

        # Actor
        self.optimizerActorCriticEval.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        # 梯度裁剪，以免爆炸
        # torch.nn.utils.clip_grad_norm(actor_network.parameters(),0.5)
        self.optimizerActorCriticEval.step()

        del self.rewards[:]
        del self.saved_log_probs[:]

        # print(list(self.actorCriticEval.parameters()))

    # step train
    def trainCriticTD(self):
        r = self.rewards[-1]

        _, futureVal = self.actorCriticTarget(
            torch.tensor(self.nextState).float().to(self.device)
        )
        val = r + futureVal
        target = val.detach()
        _, predict = self.actorCriticEval(
            torch.tensor(self.states[-1]).float().to(self.device)
        )
        # print(predict, futureVal)

        self.optimizerActorCriticEval.zero_grad()
        lossFun = torch.nn.MSELoss()
        loss = lossFun(target, predict)
        loss.backward()
        # 梯度裁剪，以免爆炸
        # torch.nn.utils.clip_grad_norm(actor_network.parameters(),0.5)
        self.optimizerActorCriticEval.step()

        # print(list(self.actorCriticEval.parameters()))

    def trainCriticMC(self):
        R = 0

        for r, s in zip(self.rewards[::-1], self.states[::-1]):
            R = r + R
            target = torch.tensor(R).float()

            _, predict = self.actorCriticEval(torch.tensor(s).float().to(self.device))

            self.optimizerActorCriticEval.zero_grad()
            lossFun = torch.nn.MSELoss()
            loss = lossFun(target, predict)
            loss.backward()
            # 梯度裁剪，以免爆炸
            # torch.nn.utils.clip_grad_norm(actor_network.parameters(),0.5)
            self.optimizerActorCriticEval.step()
        # print(predict.item(), target.item())

        # print(list(self.actorCriticEval.parameters()))

    # 逐步更新 target NN
    def updateTarget(self):
        for paramEval, paramTarget in zip(
            self.actorCriticEval.parameters(), self.actorCriticTarget.parameters()
        ):
            paramTarget.data = paramTarget.data + 0.1 * (
                paramEval.data - paramTarget.data
            )


class ActorCriticNet(torch.nn.Module):
    def __init__(self, n_actions, n_features):
        super(ActorCriticNet, self).__init__()
        # 定義每層用什麼樣的形式
        self.fc1 = torch.nn.Linear(n_features, 10)
        self.fc2 = torch.nn.Linear(10, n_actions)  # Prob of Left

        self.fc3 = torch.nn.Linear(n_features, 10)
        self.fc4 = torch.nn.Linear(10, 1)  # Prob of Left

    def forward(self, x):  # 這同時也是 Module 中的 forward 功能
        # 正向傳播輸入值, 神經網絡分析出輸出值
        x_a = self.fc1(x)
        x_a = F.relu6(x_a)
        action = F.softmax(self.fc2(x_a))

        x_v = self.fc3(x)
        x_v = F.relu6(x_v)
        val = self.fc4(x_v)

        return action, val
