import torch
import torch.nn.functional as F
import numpy as np

from Gym.models.DDPGBase import DDPGBase


class DDPG(DDPGBase):
    def __init__(
        self,
        device,
        n_actions,
        n_features,
        img_shape,
        learning_rate=0.01,
        gamma=0.9,
        tau=0.001,
        noiseStart=2,
        noiseEnd=0.2,
        noiseDecayFreq=10 ** 5,
        updateTargetFreq=10000,
        mSize=10000,
        batchSize=200,
        startTrainSize=100,
        transforms=None,
    ):
        actorCriticEval = ActorCriticNet(img_shape, n_actions, n_features)
        actorCriticTarget = ActorCriticNet(img_shape, n_actions, n_features)

        # optimizer 是訓練的工具
        # 傳入 net 的所有參數, 學習率
        optimizerCritic = torch.optim.Adam(
            actorCriticEval.critic.parameters(), lr=learning_rate
        )
        optimizerActor = torch.optim.Adam(
            actorCriticEval.actor.parameters(), lr=learning_rate
        )

        super().__init__(
            device=device,
            actorCriticEval=actorCriticEval,
            actorCriticTarget=actorCriticTarget,
            optimizerCritic=optimizerCritic,
            optimizerActor=optimizerActor,
            n_actions=n_actions,
            learning_rate=learning_rate,
            gamma=gamma,
            tau=tau,
            noiseStart=noiseStart,
            noiseEnd=noiseEnd,
            noiseDecayFreq=noiseDecayFreq,
            updateTargetFreq=updateTargetFreq,
            mSize=mSize,
            batchSize=batchSize,
            startTrainSize=startTrainSize,
            transforms=transforms,
        )

    def choose_action(self, state):
        action = super().choose_action(state)

        action = F.softmax(torch.from_numpy(action), dim=1).numpy()

        a = np.argmax(action, axis=1)[0]

        return action, a


class ActorNet(torch.nn.Module):
    def __init__(self, n_actions, n_features):
        super(ActorNet, self).__init__()
        # 定義每層用什麼樣的形式
        self.fc1 = torch.nn.Linear(n_features, n_actions)

    def forward(self, state):
        x = F.softmax(self.fc1(state), dim=1)

        return x


class CriticNet(torch.nn.Module):
    def __init__(self, n_actions, n_features):
        super(CriticNet, self).__init__()
        # 定義每層用什麼樣的形式
        self.fcVal1_s = torch.nn.Linear(n_features, 64)

        self.fcVal1_a = torch.nn.Linear(n_actions, 64)

        self.fcVal1 = torch.nn.Linear(128, 1)

    def forward(self, x, a):  # 這同時也是 Module 中的 forward 功能
        # 正向傳播輸入值, 神經網絡分析出輸出值
        x_v = F.relu(self.fcVal1_s(x))

        x_a = F.relu(self.fcVal1_a(a))

        x = torch.cat((x_v, x_a), dim=1)
        qVal = self.fcVal1(x)

        return qVal


class CNN(torch.nn.Module):
    def __init__(self, img_shape, n_features):  # chan x h x w
        super(CNN, self).__init__()
        in_channels = img_shape[2]
        h = img_shape[0]
        w = img_shape[1]

        kernel_size = 8
        stride = 4
        padding = 0
        self.conv1 = torch.nn.Conv2d(
            in_channels, 32, kernel_size=kernel_size, stride=stride, padding=padding
        )
        h = (h + padding * 2 - kernel_size) // stride + 1
        w = (w + padding * 2 - kernel_size) // stride + 1

        # self.pool1 = torch.nn.MaxPool2d(2)  # 32 x (h-2)//2 x (w-2)//2
        # h //= 2
        # w //= 2

        kernel_size = 4
        stride = 2
        padding = 0
        self.conv2 = torch.nn.Conv2d(
            32, 64, kernel_size=kernel_size, stride=stride, padding=padding
        )
        h = (h + padding * 2 - kernel_size) // stride + 1
        w = (w + padding * 2 - kernel_size) // stride + 1

        # self.pool2 = torch.nn.MaxPool2d(2)  # 64 x ((h-2)//2-2)//2 x ((w-2)//2-2)//2
        # h //= 2
        # w //= 2

        kernel_size = 3
        stride = 1
        padding = 0
        self.conv3 = torch.nn.Conv2d(
            64, 64, kernel_size=kernel_size, stride=stride, padding=padding
        )
        h = (h + padding * 2 - kernel_size) // stride + 1
        w = (w + padding * 2 - kernel_size) // stride + 1

        self.fc1 = torch.nn.Linear(64 * h * w, 512)
        self.fc2 = torch.nn.Linear(512, n_features)

        # self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x):
        # x = self.pool1(F.relu(self.conv1(x)))
        # x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.shape[0], -1)
        # x = self.dropout(x)
        x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        x = self.fc2(x)

        return x


class ActorCriticNet(torch.nn.Module):
    def __init__(self, img_shape, n_actions, n_features):
        super(ActorCriticNet, self).__init__()
        # 定義每層用什麼樣的形式
        self.img_featuresActor = CNN(img_shape, n_features)
        self.img_featuresCritic = CNN(img_shape, n_features)
        self.actor = ActorNet(n_actions, n_features)
        self.critic = CriticNet(n_actions, n_features)

    def forward(self, x):  # 這同時也是 Module 中的 forward 功能
        # 正向傳播輸入值, 神經網絡分析出輸出值
        x1 = self.img_featuresActor(x)
        action = self.actor(x1)

        x2 = self.img_featuresCritic(x)
        qVal = self.critic(x2, action)

        return action, qVal

    def action(self, x):
        x = self.img_featuresActor(x)
        action = self.actor(x)

        return action

    def criticism(self, x, a):
        x = self.img_featuresCritic(x)
        qVal = self.critic(x, a)

        return qVal
