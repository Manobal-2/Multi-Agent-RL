#!/usr/bin/env python
# coding: utf-8

# In[2]:


import gymnasium as gym
import numpy as np

from gymnasium import spaces
from gymnasium.utils.env_checker import check_env

class ToyTradingEnv(gym.Env):

    def __init__(
        self,
        episode_length=100,
        initial_price=100.0,
        daily_vol=0.01
    ):
        super().__init__()

        self.episode_length = episode_length

        self.initial_price = initial_price

        self.daily_vol = daily_vol

        self.observation_space = spaces.Box(
            low=np.array(
                [-np.inf, -1.0],
                dtype=np.float32
            ),
            high=np.array(
                [np.inf, 1.0],
                dtype=np.float32
            ),
            dtype=np.float32
        )

        self.action_space = spaces.Discrete(3)

        self.price = None
        self.position = None
        self.step_count = None

    def reset(
        self,
        seed=None,
        options=None
    ):

        super().reset(seed=seed)

        self.price = self.initial_price

        self.position = 0.0

        self.step_count = 0

        obs = np.array(
            [0.0, self.position],
            dtype=np.float32
        )

        return obs, {}

    def step(self, action):

        price_return = float(
            self.np_random.normal(
                0.0,
                self.daily_vol
            )
        )

        self.price *= (
            1.0 + price_return
        )

        action_to_position = {
            0: 1.0,
            1: self.position,
            2: -1.0
        }

        new_position = action_to_position[
            int(action)
        ]

        reward = (
            self.position
            * price_return
            * 100.0
        )

        self.position = new_position

        self.step_count += 1

        terminated = (
            self.step_count
            >= self.episode_length
        )

        truncated = False

        obs = np.array(
            [
                price_return,
                self.position
            ],
            dtype=np.float32
        )

        return (
            obs,
            reward,
            terminated,
            truncated,
            {}
        )

    def render(self):

        print(
            f"Step {self.step_count:3d} | "
            f"Price {self.price:8.2f} | "
            f"Position {self.position:+.0f}"
        )


def run_random_agent(
    n_episodes=10
):

    env = ToyTradingEnv()

    total_rewards = []

    for ep in range(n_episodes):

        obs, info = env.reset()

        total_reward = 0.0

        done = False

        while not done:

            action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(
                action
            )

            total_reward += reward

            done = (
                terminated
                or truncated
            )

        total_rewards.append(
            total_reward
        )

        print(
            f"Episode {ep + 1:2d}: "
            f"total reward = {total_reward:7.2f}"
        )

    mean_reward = np.mean(
        total_rewards
    )

    print(
        f"\nMean over {n_episodes} episodes: "
        f"{mean_reward:.2f}"
    )

    env.close()

    return total_rewards


check_env(
    ToyTradingEnv(),
    warn=True
)

print("Environment check passed")

random_rewards = run_random_agent()


# In[ ]:




