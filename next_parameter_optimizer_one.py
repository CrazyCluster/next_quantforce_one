
"""
bayes_param_optimizer.py

Bayesian parameter optimizer using Optuna + multiprocessing for multi-symbol evaluation.

- Imports your strategy module `strategy_tuneable` (compute_indicators_ta, apply_strategy_tuneable).
- Optimizes tunable parameters using Optuna (TPE sampler).
- Evaluates each candidate across a list of symbols in parallel (multiprocessing).
- Saves best params (JSON) and a CSV of trial results.

Usage:
    python bayes_param_optimizer.py --symbols AAPL MSFT NVDA AMZN --trials 120 --workers 6

Tune the defaults as you wish.
"""
import requests
import argparse
import json
import math
import time
from functools import partial
from multiprocessing import Pool
from typing import Dict, List
from bs4 import BeautifulSoup
import numpy as np
import optuna, ta
import pandas as pd
import yfinance as yf


def fetch_info():
    try:
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:101.0) Gecko/20100101 Firefox/101.0',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.5',
        }

        #  Send GET request
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, "html.parser")

        #  Get the symbols table
        tables = soup.find_all('table')

        #  #  Convert table to dataframe
        df = pd.read_html(str(tables))[4]

        #  Rename symbol
        df.rename(columns={"Ticker": "Symbol"}, inplace=True)

        return df['Symbol'].to_list()
    except Exception as e:
        print('Error loading data: ', e)
        return None


def compute_indicators_ta(df: pd.DataFrame) -> pd.DataFrame:
    'Compute indicators used by the tuneable strategy.'
    df = df.copy()
    if len(df) < 60:
        return df.assign(
            ATR=np.nan,
            RSI=np.nan,
            MACD=np.nan,
            MACD_signal=np.nan,
            MACD_hist=np.nan,
            MA20=np.nan,
            MA50=np.nan,
            ADX=np.nan,
            High_20=np.nan,
            IRG=np.nan,
            Vol20=np.nan,
            ATR_mean50=np.nan
        )

    try:
        df['ATR'] = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14).average_true_range()
    except Exception:
        df['ATR'] = np.nan

    try:
        df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
    except Exception:
        df['RSI'] = np.nan

    try:
        macd = ta.trend.MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_hist'] = macd.macd_diff()
    except Exception:
        df[['MACD', 'MACD_signal', 'MACD_hist']] = np.nan

    try:
        df['MA20'] = ta.trend.SMAIndicator(close=df['Close'], window=20).sma_indicator()
        df['MA50'] = ta.trend.SMAIndicator(close=df['Close'], window=50).sma_indicator()
    except Exception:
        df[['MA20', 'MA50']] = np.nan

    try:
        df['ADX'] = ta.trend.ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14).adx()
    except Exception:
        df['ADX'] = np.nan

    try:
        df['High_20'] = df['High'].rolling(window=20).max().shift(1)
    except Exception:
        df['High_20'] = np.nan

    try:
        df['IRG'] = (df['Close'] - df['MA20']) / df['ATR']
    except Exception:
        df['IRG'] = np.nan

    try:
        df['Vol20'] = df['Volume'].rolling(20).mean()
    except Exception:
        df['Vol20'] = np.nan

    try:
        df['ATR_mean50'] = df['ATR'].rolling(50).mean()
    except Exception:
        df['ATR_mean50'] = np.nan

    return df


def apply_strategy(df: pd.DataFrame,
                            sl_mult: float = 2.5,
                            tp_mult: float = 3.5,
                            min_momentum_pct: float = 0.02,
                            momentum_days: int = 7,
                            volume_factor: float = 1.2,
                            adx_threshold: float = 25.0,
                            irg_threshold: float = 1.0,
                            require_macd: bool = True,
                            require_ma50: bool = True,
                            require_rsi_gt: float = 30.0,
                            allow_partial_signals: bool = False,
                            **kwargs) -> pd.DataFrame:
    'Tunable buy/sell strategy.'
    df = df.copy()
    needed = ['ATR', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist', 'MA20', 'MA50', 'ADX', 'High_20', 'IRG', 'Vol20']
    if not all(c in df.columns for c in needed):
        df = compute_indicators_ta(df)

    df.loc[:, 'Buy'] = False
    df.loc[:, 'Sell'] = False
    df.loc[:, 'Position'] = 0
    df.loc[:, 'Entry_price'] = np.nan
    df.loc[:, 'Stop_Loss'] = np.nan
    df.loc[:, 'Take_Profit'] = np.nan
    df.loc[:, 'Exit_Reason'] = None

    in_position = False
    entry_price = None
    current_SL = None
    current_TP = None

    start = max(60, momentum_days + 2)
    for i in range(start, len(df)):
        row = df.iloc[i]

        if i >= momentum_days:
            past_close = df['Close'].iloc[i - momentum_days]
            momentum_pct = (row['Close'] - past_close) / past_close if past_close != 0 else 0.0
        else:
            momentum_pct = 0.0

        avg_vol = df['Vol20'].iloc[i] if not pd.isna(df['Vol20'].iloc[i]) else 0.0
        vol_ok = (avg_vol > 0) and (row['Volume'] > volume_factor * avg_vol)

        breakout_ok = (not pd.isna(df['High_20'].iloc[i])) and (row['Close'] > df['High_20'].iloc[i])
        irg_ok = (not pd.isna(row['IRG'])) and (row['IRG'] >= irg_threshold)
        adx_ok = (not pd.isna(row['ADX'])) and (row['ADX'] >= adx_threshold)

        ma50_ok = True if not require_ma50 else ((not pd.isna(row.get('MA50'))) and (row['Close'] > row.get('MA50')))
        macd_ok = True if not require_macd else ((not pd.isna(row.get('MACD'))) and (not pd.isna(row.get('MACD_signal'))) and (row['MACD'] > row.get('MACD_signal')))
        rsi_ok = (row.get('RSI', 0) > require_rsi_gt)
        momentum_ok = (momentum_pct >= min_momentum_pct)

        if allow_partial_signals:
            checks = [breakout_ok, ma50_ok, adx_ok, macd_ok, irg_ok, momentum_ok, vol_ok, rsi_ok]
            buy_cond = sum(bool(x) for x in checks) >= 4
        else:
            buy_cond = breakout_ok and ma50_ok and adx_ok and macd_ok and irg_ok and momentum_ok and vol_ok and rsi_ok

        if (not in_position) and buy_cond:
            in_position = True
            entry_price = row['Close']
            atr = row['ATR'] if not pd.isna(row['ATR']) else 0.0
            current_SL = entry_price - sl_mult * atr
            current_TP = entry_price + tp_mult * atr

            df.at[df.index[i], 'Buy'] = True
            df.at[df.index[i], 'Entry_price'] = entry_price
            df.at[df.index[i], 'Stop_Loss'] = current_SL
            df.at[df.index[i], 'Take_Profit'] = current_TP
            df.at[df.index[i], 'Position'] = 1
            continue

        if in_position:
            price = row['Close']
            exit_reason = None

            if (not pd.isna(current_SL)) and (price <= current_SL):
                exit_reason = 'SL'
            elif (not pd.isna(current_TP)) and (price >= current_TP):
                exit_reason = 'TP'
            elif (not pd.isna(row.get('MA20'))) and (price < row.get('MA20')):
                exit_reason = 'MA20_break'
            elif (not pd.isna(row.get('MACD'))) and (not pd.isna(row.get('MACD_signal'))) and (row.get('MACD') < row.get('MACD_signal')):
                exit_reason = 'MACD_loss'
            elif (not pd.isna(row.get('ADX'))) and (row.get('ADX') < max(12, adx_threshold * 0.6)):
                exit_reason = 'ADX_fall'

            if exit_reason:
                df.at[df.index[i], 'Sell'] = True
                df.at[df.index[i], 'Stop_Loss'] = current_SL
                df.at[df.index[i], 'Take_Profit'] = current_TP
                df.at[df.index[i], 'Exit_Reason'] = exit_reason
                df.at[df.index[i], 'Position'] = 0

                in_position = False
                entry_price = None
                current_SL = None
                current_TP = None
            else:
                df.at[df.index[i], 'Position'] = 1
        else:
            df.at[df.index[i], 'Position'] = 0

    return df

# ---------------------------
# Helper: evaluate one symbol with given params
# ---------------------------
def evaluate_symbol(symbol: str, params: Dict, period: str = "2y") -> Dict:
    """Download data and evaluate apply_strategy_tuneable for one symbol.
    Returns metrics dict: sharpe, max_drawdown, trades, total_return.
    """
    try:
        df = yf.download(symbol, period=period, auto_adjust=True, progress=False)
        if df.empty:
            return {"sharpe": 0.0, "max_drawdown": 0.0, "trades": 0, "total_return": 0.0}

        if isinstance(df.columns, pd.MultiIndex):
            try:
                # choose 'Close','High','Low','Volume' - if nested, take level 1
                first = df.columns.levels[1][0]
                df = df.xs(first, axis=1, level=1)
            except Exception:
                df.columns = [str(c) for c in df.columns]

        # ensure indicators
        df = compute_indicators_ta(df)

        # convert boolean-like params
        params_local = params.copy()
        if "allow_partial_signals" in params_local:
            params_local["allow_partial_signals"] = bool(round(params_local["allow_partial_signals"]))
        if "require_macd" in params_local:
            params_local["require_macd"] = bool(round(params_local["require_macd"]))
        if "require_ma50" in params_local:
            params_local["require_ma50"] = bool(round(params_local["require_ma50"]))

        df_res = apply_strategy(df, **params_local)

        # build trade returns
        returns = []
        in_pos = False
        entry = None
        for _, r in df_res.iterrows():
            if (not in_pos) and r.get("Buy", False):
                in_pos = True
                entry = float(r["Close"])
            elif in_pos and r.get("Sell", False):
                exitp = float(r["Close"])
                if entry and entry > 0:
                    returns.append((exitp / entry) - 1.0)
                in_pos = False
                entry = None

        if len(returns) == 0:
            return {"sharpe": 0.0, "max_drawdown": 0.0, "trades": 0, "total_return": 0.0}

        arr = np.array(returns)
        total_return = float(np.prod(1 + arr) - 1)
        if arr.std() > 0:
            sharpe = float((arr.mean() / arr.std()) * math.sqrt(252))
        else:
            sharpe = 0.0

        equity = np.cumprod(1 + arr)
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak
        max_dd = float(abs(dd.min())) if len(dd) > 0 else 0.0

        return {"sharpe": sharpe, "max_drawdown": max_dd, "trades": len(arr), "total_return": total_return}

    except Exception as e:
        # safe fallback
        return {"sharpe": 0.0, "max_drawdown": 0.0, "trades": 0, "total_return": 0.0}


# ---------------------------
# Objective aggregator (multi-symbol)
# ---------------------------
def evaluate_params_on_universe(symbols: List[str], params: Dict, period: str, workers: int = 4) -> Dict:
    """Evaluate params across symbols in parallel, aggregate metrics and compute a scalar fitness."""
    if workers is None or workers <= 1:
        results = [evaluate_symbol(s, params, period=period) for s in symbols]
    else:
        with Pool(processes=workers) as pool:
            fn = partial(evaluate_symbol, params=params, period=period)
            results = pool.map(fn, symbols)

    # aggregate
    sharpe_mean = float(np.mean([r["sharpe"] for r in results]))
    maxdd_mean = float(np.mean([r["max_drawdown"] for r in results]))
    trades_total = int(np.sum([r["trades"] for r in results]))
    total_return_mean = float(np.mean([r["total_return"] for r in results]))

    # fitness function: maximize sharpe, penalize drawdown, penalize too-few-trades
    # tune these weights as needed
    trade_penalty = 0.1 if trades_total < max(1, len(symbols)) else 0.0
    fitness = sharpe_mean - maxdd_mean - trade_penalty

    return {
        "fitness": float(fitness),
        "sharpe_mean": sharpe_mean,
        "max_drawdown_mean": maxdd_mean,
        "trades": trades_total,
        "total_return_mean": total_return_mean
    }


# ---------------------------
# Optuna objective
# ---------------------------
def make_objective(symbols: List[str], period: str, workers: int):
    def objective(trial: optuna.Trial):
        # Define parameter search space
        adx_threshold = trial.suggest_int("adx_threshold", 12, 40)
        irg_threshold = trial.suggest_float("irg_threshold", 0.2, 2.0, step=0.05)
        min_momentum_pct = trial.suggest_float("min_momentum_pct", 0.005, 0.05, step=0.001)
        momentum_days = trial.suggest_int("momentum_days", 3, 14)
        volume_factor = trial.suggest_float("volume_factor", 1.0, 2.0, step=0.05)
        sl_mult = trial.suggest_float("sl_mult", 1.0, 4.0, step=0.1)
        tp_mult = trial.suggest_float("tp_mult", 1.5, 6.0, step=0.1)
        allow_partial_signals = trial.suggest_categorical("allow_partial_signals", [0, 1])
        require_macd = trial.suggest_categorical("require_macd", [0, 1])
        require_ma50 = trial.suggest_categorical("require_ma50", [0, 1])
        require_rsi_gt = trial.suggest_int("require_rsi_gt", 10, 50)

        params = {
            "adx_threshold": float(adx_threshold),
            "irg_threshold": float(irg_threshold),
            "min_momentum_pct": float(min_momentum_pct),
            "momentum_days": int(momentum_days),
            "volume_factor": float(volume_factor),
            "sl_mult": float(sl_mult),
            "tp_mult": float(tp_mult),
            "allow_partial_signals": float(allow_partial_signals),
            "require_macd": float(require_macd),
            "require_ma50": float(require_ma50),
            "require_rsi_gt": float(require_rsi_gt),
        }

        # evaluate across universe
        agg = evaluate_params_on_universe(symbols, params, period=period, workers=workers)

        # Optuna maximizes objective by using return value; return fitness
        trial.set_user_attr("sharpe_mean", agg["sharpe_mean"])
        trial.set_user_attr("max_drawdown_mean", agg["max_drawdown_mean"])
        trial.set_user_attr("trades", agg["trades"])
        trial.set_user_attr("total_return_mean", agg["total_return_mean"])
        trial.set_user_attr("params", params)

        # You can also ask Optuna to prune based on some intermediate result (not done here)
        return agg["fitness"]

    return objective


# ---------------------------
# Runner
# ---------------------------
def run_optimize(symbols: List[str], n_trials: int = 100, period: str = "2y", workers: int = 4, study_name: str = None, storage: str = None):
    sampler = optuna.samplers.TPESampler(seed=42)
    if storage:
        study = optuna.create_study(direction="maximize", sampler=sampler, study_name=study_name, storage=storage, load_if_exists=True)
    else:
        study = optuna.create_study(direction="maximize", sampler=sampler, study_name=study_name)

    objective = make_objective(symbols, period, workers)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # collect best
    best_trial = study.best_trial
    best_params = best_trial.user_attrs.get("params", None)
    if not best_params:
        # fallback: reconstruct from trial params
        best_params = {k: v for k, v in best_trial.params.items()}

    # write outputs
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_json = "optuna_best_parameters.json"
    with open(out_json, "w") as fh:
        json.dump({"fitness": best_trial.value, "params": best_params, "trial_number": best_trial.number}, fh, indent=2)

    # write trials to CSV
    rows = []
    for t in study.trials:
        row = {"trial_number": t.number, "value": t.value}
        row.update(t.params)
        row.update(t.user_attrs)
        rows.append(row)
    df = pd.DataFrame(rows)
    out_csv = f"optuna_trials.csv"
    df.to_csv(out_csv, index=False)

    print(f"Best fitness: {best_trial.value}")
    print(f"Best params saved to {out_json}")
    print(f"Trials saved to {out_csv}")
    return study, best_trial, out_json, out_csv


# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="+", default=["AAPL", "MSFT", "NVDA", "AMZN"], help="Symbol list")
    p.add_argument("--trials", type=int, default=80, help="Number of Optuna trials")
    p.add_argument("--workers", type=int, default=4, help="Parallel workers for symbol evaluation (multiprocessing Pool size)")
    p.add_argument("--period", type=str, default="2y", help="yfinance history period")
    p.add_argument("--storage", type=str, default=None, help="Optuna storage (sqlite DB path). e.g. sqlite:///optuna.db")
    args = p.parse_args()
    nasdaq_symbols = fetch_info()
    run_optimize(symbols=nasdaq_symbols, n_trials=args.trials, period=args.period, workers=args.workers, storage=args.storage)
    # run_optimize(symbols=args.symbols, n_trials=args.trials, period=args.period, workers=args.workers, storage=args.storage)
