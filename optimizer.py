"""NIFTY-50 portfolio optimizer.

Builds long-only mean-variance portfolios on 2013-2019 prices, then tests them
out of sample on Jan 2020 - Oct 2022 against an equal-weight portfolio.

    pip install -r requirements.txt
    python optimizer.py
"""
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"
SOURCE = "https://github.com/Ram9219/NIFTY-50-Stock-Market-Data-2000---2022-"

TRAIN = ("2013-01-01", "2019-12-31")
TEST = ("2020-01-01", "2022-12-31")
MAX_WEIGHT = 0.15     # no single stock above 15%
RISK_FREE = 0.065
DAYS = 252

# Level shifts in the source that are data errors, not real moves (see README)
FIXES = {"TATASTEEL": "2022-06-16", "HDFC": "2015-12-28", "ADANIENT": "2015-06-03"}


def load_prices():
    if not RAW.exists():
        subprocess.run(["git", "clone", "--depth", "1", SOURCE, str(RAW)], check=True)
    series = {}
    for f in sorted(RAW.glob("*.csv")):
        d = pd.read_csv(f)
        fmt = "%d-%m-%Y" if "-" in d.Date[0][:3] else "%Y-%m-%d"
        series[f.stem] = d.assign(Date=pd.to_datetime(d.Date, format=fmt)).set_index("Date")["Adj Close"]
    p = pd.DataFrame(series).sort_index()
    for t, date in FIXES.items():
        i = p.index.get_loc(pd.Timestamp(date))
        p.loc[p.index[:i], t] *= p[t].iloc[i] / p[t].iloc[i - 1]
    return p


def annual_stats(w, mu, cov):
    ret, vol = w @ mu, np.sqrt(w @ cov @ w)
    return ret, vol, (ret - RISK_FREE) / vol


def optimise(mu, cov, objective, target=None):
    n = len(mu)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    if target is not None:
        cons.append({"type": "eq", "fun": lambda w: w @ mu - target})
    funcs = {"min_vol": lambda w: w @ cov @ w,
             "max_sharpe": lambda w: -annual_stats(w, mu, cov)[2]}
    res = minimize(funcs[objective], np.full(n, 1 / n), method="SLSQP",
                   bounds=[(0, MAX_WEIGHT)] * n, constraints=cons, options={"maxiter": 500})
    return res.x


def frontier(mu, cov, points=40):
    lo = annual_stats(optimise(mu, cov, "min_vol"), mu, cov)[0]
    hi = np.sort(mu)[-int(1 / MAX_WEIGHT):].mean()   # best achievable with the weight cap
    out = []
    for t in np.linspace(lo, hi * 0.995, points):
        w = optimise(mu, cov, "min_vol", target=t)
        out.append(annual_stats(w, mu, cov)[:2])
    return np.array(out)


def performance(returns):
    growth = (1 + returns).cumprod()
    years = len(returns) / DAYS
    cagr = growth.iloc[-1] ** (1 / years) - 1
    vol = returns.std() * np.sqrt(DAYS)
    return {"CAGR": cagr, "Volatility": vol, "Sharpe": (cagr - RISK_FREE) / vol,
            "Max drawdown": (growth / growth.cummax() - 1).min(), "Final value of 1": growth.iloc[-1]}


def main():
    prices = load_prices()
    prices = prices.loc[:, prices.loc[TRAIN[0]:].iloc[0].notna()]   # stocks listed for the whole period
    rets = prices.pct_change(fill_method=None)
    train, test = rets.loc[TRAIN[0]:TRAIN[1]].dropna(), rets.loc[TEST[0]:TEST[1]].dropna()
    mu, cov = train.mean().values * DAYS, train.cov().values * DAYS
    tickers = train.columns

    portfolios = {"Equal weight": np.full(len(tickers), 1 / len(tickers)),
                  "Min volatility": optimise(mu, cov, "min_vol"),
                  "Max Sharpe": optimise(mu, cov, "max_sharpe")}

    OUT.mkdir(exist_ok=True)
    weights = pd.DataFrame(portfolios, index=tickers)
    weights[weights > 1e-4].round(4).to_csv(OUT / "weights.csv")

    # buy-and-hold from the first test day: weights drift with prices, no rebalancing
    test_prices = prices.loc[test.index[0] - pd.Timedelta(days=7):TEST[1]].ffill().loc[test.index[0]:]
    results, curves = {}, {}
    for name, w in portfolios.items():
        value = (test_prices / test_prices.iloc[0]) @ w
        curves[name] = value
        r = value.pct_change().dropna()
        results[name] = {"in_sample": dict(zip(["Return", "Volatility", "Sharpe"], annual_stats(w, mu, cov))),
                         "out_of_sample": performance(r),
                         "holdings": int((w > 1e-4).sum())}

    table = pd.DataFrame({k: v["out_of_sample"] for k, v in results.items()}).T
    table["Holdings"] = [v["holdings"] for v in results.values()]
    table.round(4).to_csv(OUT / "out_of_sample.csv")
    json.dump(results, open(OUT / "results.json", "w"), indent=2, default=float)

    # charts
    fr = frontier(mu, cov)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(np.sqrt(np.diag(cov)) * 100, mu * 100, s=14, color="#9ca3af", label="Individual stocks")
    ax.plot(fr[:, 1] * 100, fr[:, 0] * 100, color="#2563eb", lw=2, label=f"Efficient frontier (max {MAX_WEIGHT:.0%}/stock)")
    for name, w in portfolios.items():
        r, v, _ = annual_stats(w, mu, cov)
        ax.scatter(v * 100, r * 100, s=90, marker="*", label=name, zorder=3)
    ax.set(xlabel="Annualised volatility (%)", ylabel="Annualised return (%)", title="Efficient frontier, 2013–2019 (in-sample)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "efficient_frontier.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for name, c in curves.items():
        ax.plot(c.index, c, lw=1.6, label=name)
    ax.set(ylabel="Value of ₹1", title="Out-of-sample: Jan 2020 – Oct 2022 (buy and hold)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "backtest.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    top = weights[["Min volatility", "Max Sharpe"]]
    top = top[(top > 0.01).any(axis=1)].sort_values("Max Sharpe")
    top.plot.barh(ax=ax, color=["#16a34a", "#dc2626"])
    ax.set(xlabel="Weight", title="Optimised portfolio weights")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "weights.png", dpi=150); plt.close(fig)

    pd.set_option("display.width", 160)
    print(f"{len(tickers)} stocks | train {train.index[0].date()} – {train.index[-1].date()} | "
          f"test {test.index[0].date()} – {test.index[-1].date()}\n")
    print(table.round(3))
    print("\nNon-zero weights\n", weights[(weights.iloc[:, 1:] > 1e-4).any(axis=1)].iloc[:, 1:].round(3))


if __name__ == "__main__":
    main()
