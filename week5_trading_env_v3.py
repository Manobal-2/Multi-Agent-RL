#!/usr/bin/env python
# coding: utf-8

# In[2]:


import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque


class TradingEnvV3(gym.Env):

    def __init__(
        self,
        rho=0.5,
        episode_length=100,
        initial_price=100.0,
        daily_vol=0.01,
        history_len=5,
        transaction_cost=0.1,
    ):

        super().__init__()

        self.rho = rho

        self.episode_length = episode_length

        self.initial_price = initial_price

        self.daily_vol = daily_vol

        self.history_len = history_len

        self.transaction_cost = transaction_cost

        self.reward_scale = 100.0

        obs_dim = history_len + 1

        self.observation_space = spaces.Box(
            low=np.full(obs_dim, -100, dtype=np.float32),
            high=np.full(obs_dim, 100, dtype=np.float32),
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(3)

        self.price = None

        self.position = None

        self.step_count = None

        self.price_history = None

        self.last_return = None

    def reset(
        self,
        seed=None,
        options=None,
    ):

        super().reset(seed=seed)

        self.price = self.initial_price

        self.position = 0.0

        self.step_count = 0

        self.last_return = 0.0

        self.price_history = deque(
            [0.0] * self.history_len,
            maxlen=self.history_len,
        )

        return self.get_obs(), {}

    def generate_return(self):

        noise = self.np_random.normal(
            0.0,
            self.daily_vol,
        )

        price_return = (
            self.rho * self.last_return
            + np.sqrt(1 - self.rho ** 2) * noise
        )

        self.last_return = price_return

        return float(price_return)

    def step(
        self,
        action,
    ):

        price_return = self.generate_return()

        self.price *= (
            1.0 + price_return
        )

        self.price_history.append(
            price_return
        )

        action_to_position = {
            0: 1.0,
            1: self.position,
            2: -1.0,
        }

        new_position = action_to_position[
            int(action)
        ]

        pnl = (
            self.position
            * price_return
            * self.reward_scale
        )

        cost = (
            self.transaction_cost
            * abs(
                new_position
                - self.position
            )
        )

        reward = pnl - cost

        self.position = new_position

        self.step_count += 1
        terminated = (
        self.step_count
        >= self.episode_length
        )

        truncated = False

        info = {
            "price": self.price,
            "position": self.position,
            "return": price_return,
            "pnl": pnl,
            "cost": cost
        }

        return (
            self.get_obs(),
            reward,
            terminated,
            truncated,
            info
        )

    def get_obs(self):

        history = np.array(
            list(self.price_history),
            dtype=np.float32
        )

        observation = np.append(
            history,
            self.position
        )

        return observation.astype(
            np.float32
        )

    def render(self):

        print(
            f"Step {self.step_count} | "
            f"Price {self.price:.2f} | "
            f"Position {self.position:.1f}"
        )


if __name__ == "__main__":

    env = TradingEnvV3(
        rho=0.8
    )

    obs, info = env.reset(seed=42)

    print("Initial Observation")
    print(obs)

    total_reward = 0

    done = False

    while not done:

        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        done = terminated or truncated

    print()

    print("Episode Finished")

    print(f"Total Reward : {total_reward:.2f}")

    print(f"Final Price  : {info['price']:.2f}")

    print(f"Final Position : {info['position']}")


# In[ ]:




