import os
import yfinance as yf
import pandas as pd

os.makedirs("data", exist_ok=True)

markets = {
    "SPY": "SPY",
    "NSEI": "^NSEI"
}

for name, ticker in markets.items():

    df = yf.download(
        ticker,
        start="2015-01-01",
        end="2024-12-31",
        auto_adjust=True,
        progress=False,
        multi_level_index=False
    )

    df = df[["Close"]]

    df["Return"] = df["Close"].pct_change()

    df = df.dropna().reset_index()

    split = int(len(df) * 0.8)

    train = df.iloc[:split]
    test = df.iloc[split:]

    train.to_csv(
        f"data/{name.lower()}_train.csv",
        index=False
    )

    test.to_csv(
        f"data/{name.lower()}_test.csv",
        index=False
    )

    print(name)
    print("Rows :", len(df))
    print("Train:", len(train))
    print("Test :", len(test))
    print()