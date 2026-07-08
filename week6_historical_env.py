#!/usr/bin/env python
# coding: utf-8

# In[6]:


import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class HistoricalTradingEnv(gym.Env):

    def __init__(
        self,
        csv_file="data/spy_data.csv",
        history_len=5,
        transaction_cost=0.001
    ):

        super().__init__()

        data = pd.read_csv(csv_file)

        data.columns = [str(c).strip() for c in data.columns]

        if "Return" not in data.columns:
            raise ValueError("Return column not found.")

        data["Return"] = pd.to_numeric(
            data["Return"],
            errors="coerce"
        )

        data = data.dropna(subset=["Return"])

        self.returns = data["Return"].astype(np.float32).values

        self.history_len = history_len

        self.transaction_cost = transaction_cost

        self.current_step = history_len

        self.position = 0.0

        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Box(
            low=-5.0,
            high=5.0,
            shape=(history_len + 1,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.current_step = self.history_len

        self.position = 0.0

        return self.get_obs(), {}

    def get_obs(self):

        history = self.returns[
            self.current_step-self.history_len:
            self.current_step
        ]

        if len(history) < self.history_len:

            pad = np.zeros(
                self.history_len-len(history),
                dtype=np.float32
            )

            history = np.concatenate(
                [pad, history]
            )

        obs = np.append(
            history,
            self.position
        )

        obs = np.nan_to_num(
            obs,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        obs = np.clip(
            obs,
            -5,
            5
        )

        return obs.astype(np.float32)

    def step(self, action):

        ret = float(
            self.returns[self.current_step]
        )

        old_position = self.position

        if action == 0:
            self.position = 1.0

        elif action == 1:
            self.position = old_position

        else:
            self.position = -1.0

        reward = (
            old_position * ret
        )

        reward -= (
            self.transaction_cost *
            abs(
                self.position-old_position
            )
        )

        reward = float(
            np.nan_to_num(
                reward,
                nan=0.0,
                posinf=0.0,
                neginf=0.0
            )
        )

        reward = np.clip(
            reward,
            -1,
            1
        )

        self.current_step += 1

        terminated = (
            self.current_step >=
            len(self.returns)-1
        )

        truncated = False

        return (
            self.get_obs(),
            reward,
            terminated,
            truncated,
            {}
        )

    def render(self):

        print(
            f"Step {self.current_step} "
            f"Position {self.position:+.0f}"
        )


if __name__ == "__main__":

    env = HistoricalTradingEnv()

    obs, _ = env.reset()

    print(obs)

    total = 0

    done = False

    while not done:

        action = env.action_space.sample()

        obs, reward, terminated, truncated, _ = env.step(action)

        total += reward

        done = terminated or truncated

    print("Random Reward:", round(total, 4))


# In[ ]:




