import subprocess
import sys
import time


def run_command(args, wait=False, timeout=None, debug=False):
    try:
        if debug:
            p = subprocess.Popen(args, stdout=subprocess.PIPE)
            time.sleep(60)
            p.kill()
        elif wait:
            p = subprocess.Popen(args)
            p.wait()
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


allList = {
    "Gym": {
        "Acrobot": ["A2C", "PolicyGradient", "QLearning"],
        "CartPole": ["A2C", "DDPG", "DDPG_softmax", "PolicyGradient", "QLearning"],
        "MountainCar": ["QLearning"],
        "MountainCarContinuous": ["DDPG"],
        "Pendulum": ["A2C", "DDPG"],
        # "Pong": ["DDPG", "QLearning"],
    }
}

debug = True
for platform, envs in allList.items():
    for env, methods in envs.items():
        for method in methods:
            package = f"{platform}.{env}.{method}"
            print("=================================================")
            print(package)
            print("=================================================")
            if debug:
                run_command(["python", "-m", f"{package}.train"], debug=True)
            else:
                run_command(["python", "-m", f"{package}.train"], wait=True)
            run_command(["python", "-m", f"{package}.test"], timeout=10)
