import os
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from final_environment import FinalTradingEnv
from week6_historical_env import HistoricalTradingEnv


os.makedirs("models", exist_ok=True)


def train_baseline(train_file, name):

    env = HistoricalTradingEnv(train_file)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        gamma=0.99,
        n_steps=256,
        batch_size=64,
        seed=1,
        verbose=1
    )

    model.learn(
        total_timesteps=100000
    )

    model.save(
        f"models/baseline_{name}"
    )

    mean, std = evaluate_policy(
        model,
        env,
        n_eval_episodes=5
    )

    env.close()

    print(
        "Baseline:",
        round(mean,2),
        "+/-",
        round(std,2)
    )


def train_improved(train_file, name):

    scores = []

    for seed in [1,2,3]:

        env = FinalTradingEnv(train_file)

        model = PPO(

            "MlpPolicy",

            env,

            learning_rate=3e-4,

            gamma=0.99,

            n_steps=256,

            batch_size=64,

            seed=seed,

            verbose=1

        )

        model.learn(
            total_timesteps=100000
        )

        model.save(
            f"models/{name}_{seed}"
        )

        mean,std = evaluate_policy(
            model,
            env,
            n_eval_episodes=5
        )

        print(
            "Seed",
            seed,
            round(mean,2)
        )

        scores.append(mean)

        env.close()

    print(
        "Average:",
        round(np.mean(scores),2)
    )


if __name__ == "__main__":

    print("\n========== SPY ==========\n")

    train_baseline(
        "data/spy_train.csv",
        "spy"
    )

    train_improved(
        "data/spy_train.csv",
        "spy"
    )

    print("\n========== NSEI ==========\n")

    train_baseline(
        "data/nsei_train.csv",
        "nsei"
    )

    train_improved(
        "data/nsei_train.csv",
        "nsei"
    )