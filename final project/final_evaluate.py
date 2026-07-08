import os
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO

from final_environment import FinalTradingEnv
from metrics import summary


os.makedirs("plots", exist_ok=True)


def run_model(model, env):

    obs, _ = env.reset()

    done = False

    rewards = []

    curve = [0]

    while not done:

        action, _ = model.predict(
            obs,
            deterministic=True
        )

        obs, reward, terminated, truncated, _ = env.step(action)

        rewards.append(reward)

        curve.append(
            curve[-1] + reward
        )

        done = terminated or truncated

    return rewards, curve


def run_random(env):

    obs, _ = env.reset()

    done = False

    rewards = []

    curve = [0]

    while not done:

        action = env.action_space.sample()

        obs, reward, terminated, truncated, _ = env.step(action)

        rewards.append(reward)

        curve.append(
            curve[-1] + reward
        )

        done = terminated or truncated

    return rewards, curve


def run_buy_hold(env):

    obs, _ = env.reset()

    done = False

    rewards = []

    curve = [0]

    while not done:

        obs, reward, terminated, truncated, _ = env.step(0)

        rewards.append(reward)

        curve.append(
            curve[-1] + reward
        )

        done = terminated or truncated

    return rewards, curve


def evaluate_market(name):

    env = FinalTradingEnv(
        f"data/{name}_test.csv"
    )

    model = PPO.load(
        f"models/{name}_1"
    )

    ppo_rewards, ppo_curve = run_model(
        model,
        env
    )

    env = FinalTradingEnv(
        f"data/{name}_test.csv"
    )

    random_rewards, random_curve = run_random(
        env
    )

    env = FinalTradingEnv(
        f"data/{name}_test.csv"
    )

    hold_rewards, hold_curve = run_buy_hold(
        env
    )

    print()

    print("=" * 50)

    print(name.upper())

    print()

    print(
        "Improved PPO",
        summary(ppo_rewards)
    )

    print(
        "Random",
        summary(random_rewards)
    )

    print(
        "Buy & Hold",
        summary(hold_rewards)
    )

    plt.figure(figsize=(8,4))

    plt.plot(
        ppo_curve,
        label="Improved PPO"
    )

    plt.plot(
        random_curve,
        label="Random"
    )

    plt.plot(
        hold_curve,
        label="Buy & Hold"
    )

    plt.xlabel("Step")

    plt.ylabel("Cumulative Reward")

    plt.title(name.upper())

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        f"plots/{name}_comparison.png"
    )

    plt.close()


if __name__ == "__main__":

    evaluate_market("spy")

    evaluate_market("nsei")