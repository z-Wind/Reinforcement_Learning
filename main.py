import subprocess
import sys
import time
import os


def run_command(args, wait=False, timeout=None):
    try:
        if wait:
            p = subprocess.Popen(args)
            time.sleep(60)
            p.kill()
            # p.wait()
        elif timeout:
            p = subprocess.Popen(args, stdout=subprocess.PIPE)
            time.sleep(timeout)
            p.kill()
        else:
            p = subprocess.Popen(
                args, stdin=None, stdout=None, stderr=None, close_fds=True
            )

        (result, error) = p.communicate()

    except subprocess.CalledProcessError as e:
        sys.stderr.write(
            "common::run_command() : [ERROR]: output = %s, error code = %s\n"
            % (e.output, e.returncode)
        )

    return result


dirList = [
    "Gym/Acrobot/A2C",
    "Gym/Acrobot/PolicyGradient",
    "Gym/Acrobot/QLearning",
    #
    "Gym/CartPole/A2C",
    "Gym/CartPole/DDPG",
    "Gym/CartPole/DDPG_softmax",
    "Gym/CartPole/PolicyGradient",
    "Gym/CartPole/QLearning",
    #
    "Gym/MountainCar/QLearning",
    "Gym/MountainCarContinuous/DDPG",
    #
    "Gym/Pendulum/A2C",
    "Gym/Pendulum/DDPG",
    #
    "Gym/Pong/DDPG",
    "Gym/Pong/QLearning",
]

for d in dirList:
    print("=================================================")
    print(d)
    print("=================================================")
    run_command(["python", os.path.join(d, "train.py")], wait=True)
    run_command(["python", os.path.join(d, "test.py")], timeout=10)
