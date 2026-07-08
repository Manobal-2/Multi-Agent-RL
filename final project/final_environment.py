import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class FinalTradingEnv(gym.Env):

    def __init__(
        self,
        csv_file,
        history_len=5,
        transaction_cost=0.001
    ):

        super().__init__()

        df = pd.read_csv(csv_file)

        df["Return"] = pd.to_numeric(
            df["Return"],
            errors="coerce"
        )

        df = df.dropna().reset_index(drop=True)

        self.prices = df["Close"].to_numpy(
            dtype=np.float32
        )

        self.returns = df["Return"].to_numpy(
            dtype=np.float32
        )

        self.history_len = history_len
        self.transaction_cost = transaction_cost

        self.current_step = None
        self.position = None

        self.positions = np.array(
            [
                1.0,
                0.5,
                0.0,
                -0.5,
                -1.0
            ],
            dtype=np.float32
        )

        self.action_space = spaces.Discrete(5)

        self.observation_space = spaces.Box(
            low=-100,
            high=100,
            shape=(history_len + 3,),
            dtype=np.float32
        )

    def reset(
        self,
        seed=None,
        options=None
    ):

        super().reset(seed=seed)

        self.current_step = max(
            self.history_len,
            20
        )

        self.position = 0.0

        return self.get_obs(), {}

    def get_obs(self):

        h = self.returns[
            self.current_step-self.history_len:
            self.current_step
        ]

        vol = np.std(
            self.returns[
                self.current_step-5:
                self.current_step
            ]
        )

        ma = np.mean(
            self.prices[
                self.current_step-10:
                self.current_step
            ]
        )

        dist = (
            self.prices[self.current_step]
            - ma
        ) / ma

        momentum = np.sum(
            self.returns[
                self.current_step-10:
                self.current_step
            ]
        )

        obs = np.concatenate(
            [
                h,
                [
                    vol,
                    dist,
                    momentum
                ]
            ]
        )

        return obs.astype(
            np.float32
        )

    def step(
        self,
        action
    ):

        r = self.returns[
            self.current_step
        ]

        new_position = self.positions[
            int(action)
        ]

        pnl = (
            self.position
            * r
            * 100
        )

        cost = (
            abs(
                new_position
                - self.position
            )
            * self.transaction_cost
            * 100
        )

        reward = pnl - cost

        self.position = new_position

        self.current_step += 1

        terminated = (
            self.current_step
            >= len(self.returns)-1
        )

        truncated = False

        return (
            self.get_obs(),
            reward,
            terminated,
            truncated,
            {
                "reward": reward,
                "return": r
            }
        )

    def render(self):

        print(
            self.current_step,
            self.position
        )


if __name__ == "__main__":

    env = FinalTradingEnv(
        "data/spy_train.csv"
    )

    obs, _ = env.reset()

    done = False

    total = 0

    while not done:

        a = env.action_space.sample()

        obs, r, t, tr, _ = env.step(a)

        total += r

        done = t or tr

    print(total)