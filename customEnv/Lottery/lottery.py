import numpy as np
import random


class Lottery:
    def __init__(self):
        self.state = np.random.choice(range(1, 49), 7, replace=False)
        self.reward = 0
        self.done = False
        self.info = None

        self.time = 0
        self.MAXTIME = 1000
        self.fee = 50

    def reset(self):
        self.state = np.random.choice(range(1, 49), 7, replace=False)
        self.reward = 0
        self.done = False
        self.info = None

        return self.state

    def step(self, action):
        self.time += 1
        r = self._calReward(action) - self.fee
        self.reward += r

        if self.reward > 0 or self.time > self.MAXTIME:
            self.done = True

        self.state = np.random.choice(range(1, 49), 7, replace=False)

        return self.state, r, self.done, self.info

    def _calReward(self, action):
        n = 0
        special_n = 0
        for a in action:
            for s in self.state[:6]:
                if a == s:
                    n += 1

            if a == self.state[6]:
                special_n += 1

        if n == 6:  # 頭獎
            return 100000000
        elif n == 5 and special_n == 1:  # 貳獎
            return 500000
        elif n == 5 and special_n == 0:  # 參獎
            return 31000
        elif n == 4 and special_n == 1:  # 肆獎
            return 6000
        elif n == 4 and special_n == 0:  # 伍獎
            return 2000
        elif n == 3 and special_n == 1:  # 陸獎
            return 1000
        elif n == 2 and special_n == 1:  # 柒獎
            return 400
        elif n == 3 and special_n == 0:  # 普獎
            return 400

        return 0


if __name__ == "__main__":
    env = Lottery()

    env.state = np.array([1, 2, 3, 4, 5, 6, 7])
    assert env._calReward(random.sample([1, 2, 3, 4, 5, 6], 6)) == 100000000
    assert env._calReward(random.sample([1, 2, 3, 4, 5, 7], 6)) == 500000
    assert env._calReward(random.sample([1, 2, 3, 4, 5, 8], 6)) == 31000
    assert env._calReward(random.sample([1, 2, 3, 4, 7, 8], 6)) == 6000
    assert env._calReward(random.sample([1, 2, 3, 4, 9, 8], 6)) == 2000
    assert env._calReward(random.sample([1, 2, 7, 10, 9, 8], 6)) == 400
    assert env._calReward(random.sample([1, 2, 3, 10, 9, 8], 6)) == 400
    assert env._calReward(random.sample([1, 12, 11, 10, 9, 8], 6)) == 0

    sumR = 0
    n = 100000
    for i in range(n):
        action = np.random.choice(range(1, 49), 6, replace=False)
        _, r, done, _ = env.step(action)
        sumR += r

    print(sumR, sumR / n, 50 + sumR / n)
