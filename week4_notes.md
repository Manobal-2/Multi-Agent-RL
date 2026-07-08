# Week 4 Notes


## 1.

The PPO reward curve stayed mostly around zero and did not show a strong upward trend.

This was expected because the trading environment uses a random walk, meaning future price changes cannot be predicted from past information. The agent can learn how to interact with the environment, but it should not consistently generate large profits.


## 2.

The PPO agent reward was compared with the random agent from Week 3.

There was no major improvement because the environment does not contain a predictable pattern. This shows that PPO can optimize the given reward, but it cannot create information that does not exist.


## 3.

The transaction cost discourages changing positions too frequently.

A good policy would avoid unnecessary trades and only change position when the expected reward is higher than the cost. However, in a random environment this may also encourage the agent to stay flat.


## 4.

Adding previous returns increases the amount of information available to the agent.

However, in a true random walk, past returns do not predict future returns. The extra history may still help the agent manage positions better.


## 5.

The PPO, random, and buy-and-hold strategies should end close to each other.

This happens because the environment has no predictable advantage. A large difference would probably indicate a bias or random luck.


## 6.

When transaction cost was removed, the strategies became more similar because changing positions became free.

The PPO agent could trade more often without penalty, but it did not necessarily create better performance.


## 7.

During evaluation deterministic=True selects the highest probability action instead of sampling randomly.

During training randomness is useful because it helps exploration. During evaluation we want to measure the learned policy itself.


## 8.

The hardest part of applying RL to finance is designing the environment.

The algorithm is important, but the state and reward decide what the agent can actually learn. A badly designed environment can make the agent learn unrealistic behaviour.


## 9.

Before trusting a trading agent, I would add:

1. Real historical price data.
2. Transaction costs and market impact.
3. Portfolio information and risk measures.

These would improve the state and reward design.


## 10.

Questions for mentor:

1. How should realistic financial constraints be added to RL environments?
2. How can we detect if an agent is learning or just exploiting environment mistakes?
3. Which RL algorithms work best for noisy financial problems?