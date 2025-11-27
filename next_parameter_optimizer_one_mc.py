#!/usr/bin/env python3
"""
next_parameter_optimizer_final.py

Final corrected optimizer:
 - stable fitness (no exploding sums)
 - per-symbol metrics averaged, not summed
 - Monte-Carlo per-symbol, normalized and bounded
 - multiprocessing safe (top-level worker)
 - parquet caching for price data
 - Optuna TPE + MedianPruner
 - JSON/CSV outputs for best trial + trials

Requires:
    pip install optuna yfinance pandas numpy ta pyarrow

Usage:
    python next_parameter_optimizer_final.py --symbols AAPL MSFT NVDA AMZN --trials 80 --workers 4 --period 2y
"""

import argparse
import json
import math
import multiprocessing
import os
import time

from typing import Dict, List, Tuple, Optional
import numpy as np
import optuna
import pandas as pd
import yfinance as yf

from next_quantforce_one import fetch_info, compute_indicators_ta, apply_strategy_tuneable

# ------------------------
# Config / Defaults
# ------------------------
CACHE_DIR = "cache_prices"
os.makedirs(CACHE_DIR, exist_ok=True)
DEFAULT_PERIOD = "2y"
DEFAULT_TRIALS = 80
DEFAULT_WORKERS = max(1, multiprocessing.cpu_count() - 1)
DEFAULT_MIN_TRADES = 3
TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")

# ------------------------
# Helper: Price cache (parquet)
# ------------------------
def load_price_data_cached(symbol: str, period: str = DEFAULT_PERIOD, interval: str = "1d", max_age_hours: int = 24) -> pd.DataFrame:
    """
    Load OHLCV data for `symbol` using a local parquet cache. Thread/process safe read.
    """
    cache_file = os.path.join(CACHE_DIR, f"{symbol}_{period}_{interval}.parquet")
    # return cached if not expired
    if os.path.exists(cache_file):
        age_h = (time.time() - os.path.getmtime(cache_file)) / 3600.0
        if age_h <= max_age_hours:
            try:
                return pd.read_parquet(cache_file)
            except Exception as e:
                print(f"[WARN] Failed to read cache for {symbol}: {e}")

    # download and save (single process will write; multiple processes may race but it's acceptable)
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
    except Exception as e:
        print(f"[WARN] yfinance download failed for {symbol}: {e}")
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    try:
        df.to_parquet(cache_file)
    except Exception as e:
        print(f"[WARN] Could not save parquet for {symbol}: {e}")

    return df

# ------------------------
# Backtest single DF -> trade statistics
# ------------------------
def extract_trades_from_signals(df_signals: pd.DataFrame) -> List[float]:
    """
    Given dataframe with Buy/Sell flags and Close price, extract list of trade returns.
    A Buy opens a trade (if not currently in position). The next Sell closes it.
    Returns list of returns = (exit/entry - 1).
    """
    trade_returns = []
    in_pos = False
    entry_price = None

    # Use rows in chronological order
    for _, row in df_signals.iterrows():
        buy = bool(row.get("Buy", False))
        sell = bool(row.get("Sell", False))
        price = row.get("Close", np.nan)

        if (not in_pos) and buy:
            if not pd.isna(price):
                in_pos = True
                entry_price = float(price)
        elif in_pos and sell:
            if (entry_price is not None) and (not pd.isna(price)) and entry_price > 0:
                trade_returns.append((float(price) / float(entry_price)) - 1.0)
            in_pos = False
            entry_price = None
    return trade_returns

def backtest_df(df: pd.DataFrame, params: Dict) -> Dict:
    """
    Apply strategy (indicators expected or computed), return metrics per-symbol:
      { trades, total_return, sharpe, win_rate, max_drawdown, trade_returns }
    """
    if df is None or df.empty:
        return {"trades": 0, "total_return": 0.0, "sharpe": 0.0, "win_rate": 0.0, "max_drawdown": 0.0, "trade_returns": []}
    try:
        # compute indicators if missing
        try:
            needed = ["ATR", "MA20", "MA50", "ADX", "IRG", "Vol20", "MACD", "MACD_signal", "RSI", "High_20"]
            if not all(c in df.columns for c in needed):
                df = compute_indicators_ta(df)
        except Exception:
            # if compute_indicators_ta fails, still try to run strategy; user must ensure function exists
            df = compute_indicators_ta(df)

        res = apply_strategy_tuneable(df.copy(), **params)

        trade_returns = extract_trades_from_signals(res)
        trades = len(trade_returns)
        if trades == 0:
            return {"trades": 0, "total_return": 0.0, "sharpe": 0.0, "win_rate": 0.0, "max_drawdown": 0.0, "trade_returns": []}

        arr = np.array(trade_returns)
        # total return as compounded
        total_return = float(np.prod(1 + arr) - 1.0)
        win_rate = float((arr > 0).sum() / len(arr))
        sharpe = float((arr.mean() / arr.std()) * math.sqrt(252)) if arr.std() > 0 else 0.0

        # equity curve and max drawdown
        eq = np.cumprod(1 + arr)
        peak = np.maximum.accumulate(eq)
        dd = (peak - eq) / peak
        max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0

        return {"trades": trades, "total_return": total_return, "sharpe": sharpe, "win_rate": win_rate, "max_drawdown": max_dd, "trade_returns": trade_returns}
    except Exception as e:
        print(f"[WARN] backtest_df exception: {e}")
        return {"trades": 0, "total_return": 0.0, "sharpe": 0.0, "win_rate": 0.0, "max_drawdown": 0.0, "trade_returns": []}

# ------------------------
# Multiprocessing worker (top-level, picklable)
# ------------------------
def _eval_symbol_task(args: Tuple[str, Dict, str]) -> Dict:
    symbol, params, period = args
    try:
        df = load_price_data_cached(symbol, period=period)
        if df is None or df.empty:
            return {"symbol": symbol, "trades": 0, "total_return": 0.0, "sharpe": 0.0, "win_rate": 0.0, "max_drawdown": 0.0, "trade_returns": []}

        if isinstance(df.columns, pd.MultiIndex):
            try:
                # choose 'Close','High','Low','Volume' - if nested, take level 1
                first = df.columns.levels[1][0]
                df = df.xs(first, axis=1, level=1)
            except Exception:
                df.columns = [str(c) for c in df.columns]

        metrics = backtest_df(df, params)
        metrics["symbol"] = symbol
        return metrics
    except Exception as e:
        print(f"[WARN] _eval_symbol_task error for {symbol}: {e}")
        return {"symbol": symbol, "trades": 0, "total_return": 0.0, "sharpe": 0.0, "win_rate": 0.0, "max_drawdown": 0.0, "trade_returns": []}

# ------------------------
# Monte-Carlo (per-symbol) + normalized score
# ------------------------
def monte_carlo_from_trade_returns(trade_returns: List[float], n_sim: int = 2000, noise_std: float = 0.02) -> Optional[np.ndarray]:
    """
    Resample trade returns with replacement and add small gaussian noise.
    Returns array shape (n_sim, len(trade_returns)+1) representing equity factors starting from 1.0
    """
    if not trade_returns or len(trade_returns) < 5:
        return None
    arr = np.array(trade_returns)
    base_std = arr.std() if arr.std() > 0 else 1e-6
    m = len(arr)
    sims = np.empty((n_sim, m + 1), dtype=float)
    for i in range(n_sim):
        sampled = np.random.choice(arr, size=m, replace=True)
        noise = np.random.normal(0, noise_std * base_std, size=m)
        sampled = sampled + noise
        eq = np.empty(m + 1, dtype=float)
        eq[0] = 1.0
        for j, r in enumerate(sampled, start=1):
            eq[j] = eq[j - 1] * (1.0 + r)
        sims[i, :] = eq
    return sims

def compute_monte_carlo_score_normalized(mc_curves: np.ndarray) -> float:
    """
    Compute a bounded, normalized MC-stability score from mc_curves.
    Score range approx [-5, +5]. Positive = robust; negative = fragile.
    Computation uses median final return, worst-5% final return and worst drawdown percentiles.
    """
    if mc_curves is None or len(mc_curves) == 0:
        return 0.0  # neutral for missing MC (we won't reward)
    finals = mc_curves[:, -1]
    # final returns as factor (1.0 = no change). Convert to returns
    final_returns = finals - 1.0
    median_ret = np.median(final_returns)
    worst5_ret = np.percentile(final_returns, 5)
    # drawdowns per simulation
    maxdds = []
    for curve in mc_curves:
        peak = curve[0]
        maxdd = 0.0
        for v in curve:
            peak = max(peak, v)
            dd = (peak - v) / peak
            if dd > maxdd:
                maxdd = dd
        maxdds.append(maxdd)
    median_dd = np.median(maxdds)
    worst95_dd = np.percentile(maxdds, 95)

    # Normalize components to reasonable ranges:
    # median_ret ~ -1..+5 typically, worst5_ret lower; dd in 0..1
    # Build score: reward median_ret and worst-case, penalize dd and prob of loss
    prob_loss = float((finals < 1.0).mean())

    # scale factors tuned to make typical scores between -3..+3
    score = (
        3.0 * median_ret +        # reward typical return
        2.0 * worst5_ret -        # reward robustness in tails
        2.5 * median_dd -         # penalize median drawdown
        3.0 * worst95_dd -        # penalize extreme drawdown
        1.5 * prob_loss           # penalize probability of ending net-loss
    )

    # clip to avoid runaway magnitudes and cap
    score = float(np.clip(score, -10.0, 10.0))
    return score

# ------------------------
# Evaluate parameters across universe (averaging per-symbol metrics)
# ------------------------
def evaluate_params_on_universe(symbols: List[str], params: Dict, period: str = DEFAULT_PERIOD, workers: int = 1, min_trades_required: int = DEFAULT_MIN_TRADES, n_mc: int = 1000, verbose: bool = False) -> Dict:
    """
    Evaluate params across the symbol list, compute per-symbol metrics, MC per-symbol, then aggregate.
    Returns dictionary with averaged metrics and normalized mc_score aggregated.
    """
    if not symbols:
        return {"fitness": -10.0, "trades": 0}

    tasks = [(s, params, period) for s in symbols]
    if workers and workers > 1:
        try:
            with multiprocessing.Pool(processes=workers) as pool:
                results = pool.map(_eval_symbol_task, tasks)
        except Exception as e:
            if verbose:
                print(f"[WARN] multiprocessing failed ({e}), falling back to sequential")
            results = list(map(_eval_symbol_task, tasks))
    else:
        results = list(map(_eval_symbol_task, tasks))

    # normalize results: remove None and ensure keys
    valid_results = [r for r in results if r and isinstance(r, dict)]
    # compute per-symbol MC scores and collect only symbols with trades
    mc_scores = []
    per_symbol_metrics = []
    traded_symbols = []
    for r in valid_results:
        trades = int(r.get("trades", 0))
        if trades <= 0:
            # still include for count but not in aggregated means
            per_symbol_metrics.append(r)
            continue
        # run MC on trade returns of that symbol
        trade_returns = r.get("trade_returns", []) or []
        mc_curves = monte_carlo_from_trade_returns(trade_returns, n_sim=n_mc, noise_std=0.02) if len(trade_returns) >= 5 else None
        mc_score_sym = compute_monte_carlo_score_normalized(mc_curves)
        r["mc_score"] = mc_score_sym
        mc_scores.append(mc_score_sym)
        per_symbol_metrics.append(r)
        traded_symbols.append(r["symbol"])

    # aggregated metrics averaged across symbols that had trades
    traded = [r for r in per_symbol_metrics if r.get("trades", 0) > 0]
    trades_total = int(sum(r.get("trades", 0) for r in per_symbol_metrics))
    if len(traded) == 0:
        if verbose:
            print("[INFO] No completed trades across universe with these params.")
        return {"fitness": -1.0, "trades": 0, "per_symbol": per_symbol_metrics}

    total_return_mean = float(np.mean([r["total_return"] for r in traded]))
    sharpe_mean = float(np.mean([r["sharpe"] for r in traded]))
    win_rate_mean = float(np.mean([r["win_rate"] for r in traded]))
    maxdd_mean = float(np.mean([r["max_drawdown"] for r in traded]))
    # MC aggregated: mean of per-symbol normalized scores
    mc_score_mean = float(np.mean(mc_scores)) if mc_scores else 0.0

    # Final normalized fitness composition (bounded)
    # We also clip inputs to sensible ranges to avoid NaN/Inf
    total_return_mean = float(np.nan_to_num(total_return_mean, nan=0.0, posinf=0.0, neginf=0.0))
    sharpe_mean = float(np.nan_to_num(sharpe_mean, nan=0.0, posinf=0.0, neginf=0.0))
    win_rate_mean = float(np.nan_to_num(win_rate_mean, nan=0.0, posinf=0.0, neginf=0.0))
    maxdd_mean = float(np.nan_to_num(maxdd_mean, nan=0.0, posinf=1.0, neginf=0.0))
    mc_score_mean = float(np.nan_to_num(mc_score_mean, nan=0.0, posinf=10.0, neginf=-10.0))

    # Compose fitness: weights tuned to produce reasonable numeric scale
    fitness = (
        1.0 * total_return_mean +        # reward returns
        0.6 * sharpe_mean +              # reward risk-adjusted performance
        0.4 * win_rate_mean -            # reward consistency
        0.6 * maxdd_mean +               # penalize drawdown
        0.8 * mc_score_mean              # reward Monte-Carlo robustness
    )

    # final safety clipping
    fitness = float(np.clip(fitness, -999.0, 9999999.0))

    if verbose:
        print(f"[EVAL] symbols={len(symbols)} traded_symbols={len(traded)} trades_total={trades_total} ret_mean={total_return_mean:.4f} sharpe={sharpe_mean:.3f} winrate={win_rate_mean:.3f} maxdd={maxdd_mean:.3f} mc_score={mc_score_mean:.3f} fitness={fitness:.3f}")

    return {
        "fitness": fitness,
        "trades": trades_total,
        "total_return_mean": total_return_mean,
        "sharpe_mean": sharpe_mean,
        "win_rate_mean": win_rate_mean,
        "max_drawdown_mean": maxdd_mean,
        "mc_score_mean": mc_score_mean,
        "per_symbol": per_symbol_metrics
    }

# ------------------------
# Optuna objective builder
# ------------------------
def make_objective(symbols: List[str], period: str, workers: int, min_trades_required: int, n_mc: int, verbose: bool):
    def objective(trial: optuna.Trial):
        # search space (sensible)
        adx_threshold = trial.suggest_int("adx_threshold", 8, 30)
        irg_threshold = trial.suggest_float("irg_threshold", 0.1, 1.6, step=0.05)
        min_momentum_pct = trial.suggest_float("min_momentum_pct", 0.001, 0.05, step=0.001)
        momentum_days = trial.suggest_int("momentum_days", 3, 20)
        volume_factor = trial.suggest_float("volume_factor", 0.7, 2.0, step=0.05)
        sl_mult = trial.suggest_float("sl_mult", 1.0, 4.0, step=0.1)
        tp_mult = trial.suggest_float("tp_mult", 1.0, 6.0, step=0.1)
        allow_partial_signals = int(trial.suggest_categorical("allow_partial_signals", [0, 1]))
        require_macd = int(trial.suggest_categorical("require_macd", [0, 1]))
        require_ma50 = int(trial.suggest_categorical("require_ma50", [0, 1]))
        require_rsi_gt = int(trial.suggest_int("require_rsi_gt", 10, 60))

        params = {
            "adx_threshold": float(adx_threshold),
            "irg_threshold": float(irg_threshold),
            "min_momentum_pct": float(min_momentum_pct),
            "momentum_days": int(momentum_days),
            "volume_factor": float(volume_factor),
            "sl_mult": float(sl_mult),
            "tp_mult": float(tp_mult),
            "allow_partial_signals": allow_partial_signals,
            "require_macd": require_macd,
            "require_ma50": require_ma50,
            "require_rsi_gt": require_rsi_gt,
        }

        agg = evaluate_params_on_universe(symbols, params, period=period, workers=workers, min_trades_required=min_trades_required, n_mc=n_mc, verbose=verbose)

        trial.set_user_attr("agg", agg)
        trial.set_user_attr("params", params)
        return float(agg.get("fitness", -999.0))
    return objective

# ------------------------
# Runner (Optuna + CLI)
# ------------------------
def run_optimize(symbols: List[str], n_trials: int = DEFAULT_TRIALS, period: str = DEFAULT_PERIOD, workers: int = DEFAULT_WORKERS, storage: str = None, min_trades_required: int = DEFAULT_MIN_TRADES, n_mc: int = 1000, verbose: bool = False):
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0, interval_steps=1)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner, study_name=None, storage=storage, load_if_exists=True)
    objective = make_objective(symbols, period, workers, min_trades_required, n_mc, verbose)

    print(f"[START] Optuna: trials={n_trials}, symbols={len(symbols)}, workers={workers}, period={period}, n_mc={n_mc}")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial
    best_agg = best.user_attrs.get("agg", {})
    best_params = best.user_attrs.get("params", best.params)

    out_json = "optuna_best_parameters_mc.json"
    with open(out_json, "w") as fh:
        json.dump({"fitness": best.value, "params": best_params, "agg": best_agg, "trial_number": best.number}, fh, indent=2)

    rows = []
    for t in study.trials:
        row = {"trial_number": t.number, "value": t.value}
        row.update(t.params)
        agg = t.user_attrs.get("agg", {})
        row.update({
            "trades": agg.get("trades", None),
            "total_return_mean": agg.get("total_return_mean", None),
            "sharpe_mean": agg.get("sharpe_mean", None),
            "max_drawdown_mean": agg.get("max_drawdown_mean", None),
            "mc_score_mean": agg.get("mc_score_mean", None)
        })
        rows.append(row)
    df = pd.DataFrame(rows)
    out_csv = "optuna_trials_mc.csv"
    df.to_csv(out_csv, index=False)

    print(f"[DONE] Best fitness: {best.value}, saved {out_json}, trials saved to {out_csv}")
    return study, best, out_json, out_csv

# ------------------------
# CLI
# ------------------------
if __name__ == "__main__":
    default_symbols = ["AAPL", "MSFT", "NVDA", "AMZN"]
    nasdaq_symbols = fetch_info()
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="+", default=nasdaq_symbols, help="Symbols to evaluate")
    p.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    p.add_argument("--period", type=str, default=DEFAULT_PERIOD)
    p.add_argument("--storage", type=str, default=None, help="sqlite:///optuna.db")
    p.add_argument("--min_trades", type=int, default=DEFAULT_MIN_TRADES)
    p.add_argument("--n_mc", type=int, default=1000, help="MC sims per symbol (reduce for speed)")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    run_optimize(symbols=args.symbols, n_trials=args.trials, period=args.period, workers=args.workers, storage=args.storage, min_trades_required=args.min_trades, n_mc=args.n_mc, verbose=args.verbose)
