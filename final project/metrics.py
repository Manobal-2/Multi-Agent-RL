import numpy as np


def sharpe_ratio(rewards):

    r = np.array(
        rewards,
        dtype=float
    )

    if len(r) == 0:
        return 0.0

    if r.std() == 0:
        return 0.0

    return float(
        r.mean()
        / r.std()
        * np.sqrt(252)
    )


def max_drawdown(rewards):

    equity = np.cumsum(
        rewards
    )

    peak = np.maximum.accumulate(
        equity
    )

    drawdown = peak - equity

    return float(
        drawdown.max()
    )


def final_pnl(rewards):

    return float(
        np.sum(rewards)
    )


def summary(rewards):

    return {
        "PnL": round(
            final_pnl(rewards),
            2
        ),
        "Sharpe": round(
            sharpe_ratio(rewards),
            3
        ),
        "Max Drawdown": round(
            max_drawdown(rewards),
            2
        )
    }


if __name__ == "__main__":

    r = np.random.normal(
        0,
        1,
        200
    )

    print(
        summary(r)
    )