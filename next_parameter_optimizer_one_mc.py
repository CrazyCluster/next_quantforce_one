# monte_carlo_opt_integration.py
import os
import time
import json
import math
import requests
import multiprocessing
from bs4 import BeautifulSoup
from typing import List, Dict, Tuple
import argparse
import numpy as np
import pandas as pd
import yfinance as yf
import optuna, ta

# -------------------------
# CACHE (parquet) for price data
# -------------------------
CACHE_DIR = "cache_prices"
os.makedirs(CACHE_DIR, exist_ok=True)

def load_price_data_cached(symbol: str, period: str = "2y", interval: str = "1d", max_age_hours: int = 24):
    cache_file = os.path.join(CACHE_DIR, f"{symbol}_{period}_{interval}.parquet")
    # cache hit
    if os.path.exists(cache_file):
        age_h = (time.time() - os.path.getmtime(cache_file)) / 3600.0
        if age_h <= max_age_hours:
            try:
                return pd.read_parquet(cache_file)
            except Exception:
                pass
    # download and save
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    try:
        df.to_parquet(cache_file)
    except Exception as e:
        print("[WARN] Could not save parquet:", e)
    return df

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

# -------------------------
# backtest helper returns trade-return list and metrics
# -------------------------
def backtest_symbol_from_df(df: pd.DataFrame, params: Dict) -> Dict:
    """
    Apply strategy to df (with indicators) and extract completed trade returns.
    Returns:
      { "symbol": str(optional),
        "trade_returns": List[float],
        "trades": int,
        "total_return": float,
        "sharpe": float,
        "win_rate": float,
        "max_drawdown": float }
    """
    if df is None or df.empty:
        return {"trade_returns": [], "trades": 0, "total_return": 0.0, "sharpe": 0.0, "win_rate": 0.0, "max_drawdown": 0.0}

    # ensure indicators
    required = ["ATR","MA20","MA50","ADX","IRG","Vol20","MACD","MACD_signal","RSI","High_20"]
    if not all(c in df.columns for c in required):
        df = compute_indicators_ta(df)

    df_res = apply_strategy(df.copy(), **params)
    # collect trades: match buy -> next sell
    trade_returns = []
    in_pos = False
    entry = None
    for idx, r in df_res.iterrows():
        if (not in_pos) and r.get("Buy", False):
            in_pos = True
            entry = float(r["Close"])
        elif in_pos and r.get("Sell", False):
            exitp = float(r["Close"])
            if entry and entry > 0:
                trade_returns.append((exitp / entry) - 1.0)
            in_pos = False
            entry = None
    trades = len(trade_returns)
    if trades == 0:
        return {"trade_returns": [], "trades": 0, "total_return": 0.0, "sharpe": 0.0, "win_rate": 0.0, "max_drawdown": 0.0}
    arr = np.array(trade_returns)
    total_return = float(np.prod(1 + arr) - 1.0)
    win_rate = float((arr > 0).sum() / len(arr))
    sharpe = float((arr.mean() / arr.std()) * math.sqrt(252)) if arr.std() > 0 else 0.0
    # compute equity sequence for drawdown
    eq = np.cumprod(1 + arr)
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / peak
    max_dd = float(np.max(dd)) if len(dd)>0 else 0.0
    return {"trade_returns": trade_returns, "trades": trades, "total_return": total_return, "sharpe": sharpe, "win_rate": win_rate, "max_drawdown": max_dd}

# -------------------------
# multiprocessing top-level worker (picklable)
# -------------------------
def _eval_symbol_task(args: Tuple[str, Dict, str]):
    symbol, params, period = args
    try:
        df = load_price_data_cached(symbol, period=period)
        if df is None or df.empty:
            return {"symbol": symbol, "trade_returns": [], "trades": 0, "total_return": 0.0, "sharpe": 0.0, "win_rate": 0.0, "max_drawdown": 0.0}

        if isinstance(df.columns, pd.MultiIndex):
            try:
                # choose 'Close','High','Low','Volume' - if nested, take level 1
                first = df.columns.levels[1][0]
                df = df.xs(first, axis=1, level=1)
            except Exception:
                df.columns = [str(c) for c in df.columns]

        result = backtest_symbol_from_df(df, params)
        result["symbol"] = symbol
        return result
    except Exception:
        return {"symbol": symbol, "trade_returns": [], "trades": 0, "total_return": 0.0, "sharpe": 0.0, "win_rate": 0.0, "max_drawdown": 0.0}

# -------------------------
# Monte-Carlo simulation and score (uses trade-returns)
# -------------------------
def monte_carlo_from_trade_returns(trade_returns: List[float], n_sim: int = 2000, noise_std: float = 0.02):
    """
    Build MC equity curves by resampling trade_returns with replacement.
    Adds small gaussian noise proportional to std(trade_returns).
    Returns numpy array shape (n_sim, len(trade_returns)+1) of equity curves starting at 1.0
    """
    if not trade_returns or len(trade_returns) < 5:
        return None
    arr = np.array(trade_returns)
    base_std = arr.std() if arr.std() > 0 else 1e-6
    sims = []
    m = len(arr)
    for _ in range(n_sim):
        sampled = np.random.choice(arr, size=m, replace=True)
        noise = np.random.normal(0, noise_std * base_std, size=m)
        sampled = sampled + noise
        eq = np.empty(m+1, dtype=float)
        eq[0] = 1.0
        for i, r in enumerate(sampled, start=1):
            eq[i] = eq[i-1] * (1.0 + r)
        sims.append(eq)
    return np.array(sims)

def compute_monte_carlo_score(mc_curves: np.ndarray):
    """
    Compute robust score from MC curves.
    Returns a float (higher = better robustness).
    """
    if mc_curves is None or len(mc_curves) == 0:
        return -1.0
    finals = mc_curves[:, -1]
    # compute drawdowns for each curve
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
    finals = np.array(finals)
    maxdds = np.array(maxdds)
    median_ret = np.median(finals) - 1.0  # convert from factor to return
    worst5_ret = np.percentile(finals, 5) - 1.0
    median_dd = np.median(maxdds)
    worst95_dd = np.percentile(maxdds, 95)
    prob_loss = (finals < 1.0).mean()
    # Score: reward median ret, reward worst5 ret, penalize median/worst dd and probability of loss
    score = (1.0 * median_ret) + (0.6 * worst5_ret) - (1.0 * median_dd) - (1.5 * worst95_dd) - (1.0 * prob_loss)
    return float(score)

# -------------------------
# Core evaluate_params_on_universe with MC integrated
# -------------------------
def evaluate_params_on_universe_with_mc(symbols: List[str], params: Dict, period: str = "2y", workers: int = 1, min_trades_required: int = 1, verbose: bool = False):
    """
    Evaluate params across the symbol list, gather trade returns, compute aggregated metrics,
    run Monte-Carlo on combined trade-returns and return aggregated metrics + mc_score.
    """
    if not symbols:
        return {"fitness": -10.0, "trades": 0}

    tasks = [(s, params, period) for s in symbols]

    # parallel map
    if workers and workers > 1:
        try:
            with multiprocessing.Pool(processes=workers) as pool:
                results = pool.map(_eval_symbol_task, tasks)
        except Exception as e:
            if verbose:
                print("[WARN] multiprocessing failed:", e, "falling back to sequential")
            results = list(map(_eval_symbol_task, tasks))
    else:
        results = list(map(_eval_symbol_task, tasks))

    # collect all trade returns across symbols
    all_trade_returns = []
    per_symbol_metrics = []
    trades_total = 0
    for r in results:
        tr = r.get("trade_returns", [])
        trades_total += len(tr)
        if tr:
            all_trade_returns.extend(tr)
        per_symbol_metrics.append(r)

    if trades_total == 0:
        if verbose:
            print("[INFO] No completed trades for params on universe.")
        return {"fitness": -1.0, "trades": 0, "mc_score": -1.0}

    # aggregated metrics from symbols which had trades
    valid = [r for r in per_symbol_metrics if r.get("trades", 0) > 0]
    total_return_mean = float(np.mean([r["total_return"] for r in valid]))
    sharpe_mean = float(np.mean([r["sharpe"] for r in valid]))
    win_rate_mean = float(np.mean([r["win_rate"] for r in valid]))
    maxdd_mean = float(np.mean([r["max_drawdown"] for r in valid]))

    # Monte-Carlo on combined trade returns (resample trades)
    mc_curves = monte_carlo_from_trade_returns(all_trade_returns, n_sim=2000, noise_std=0.02)
    mc_score = compute_monte_carlo_score(mc_curves)

    # final fitness: include mc_score (scaled)
    fitness = (
        1.0 * total_return_mean +
        0.8 * sharpe_mean +
        0.5 * win_rate_mean -
        0.6 * maxdd_mean +
        1.2 * mc_score
    )

    if verbose:
        print(f"[EVAL] trades_total={trades_total} ret_mean={total_return_mean:.4f} sharpe={sharpe_mean:.3f} winrate={win_rate_mean:.3f} maxdd={maxdd_mean:.3f} mc_score={mc_score:.4f} fitness={fitness:.4f}")

    return {
        "fitness": float(fitness),
        "trades": int(trades_total),
        "total_return_mean": total_return_mean,
        "sharpe_mean": sharpe_mean,
        "win_rate_mean": win_rate_mean,
        "max_drawdown_mean": maxdd_mean,
        "mc_score": mc_score,
        "per_symbol": per_symbol_metrics
    }


# ---------------------------
# Optuna objective
# ---------------------------
def make_objective(symbols: List[str], period: str, workers: int, min_trades_required: int):
    def objective(trial: optuna.Trial):
        adx_threshold = trial.suggest_int("adx_threshold", 10, 30)
        irg_threshold = trial.suggest_float("irg_threshold", 0.10, 1.60, step=0.05)
        min_momentum_pct = trial.suggest_float("min_momentum_pct", 0.002, 0.03, step=0.001)
        momentum_days = trial.suggest_int("momentum_days", 3, 10)
        volume_factor = trial.suggest_float("volume_factor", 0.8, 1.6, step=0.05)
        sl_mult = trial.suggest_float("sl_mult", 1.0, 3.0, step=0.1)
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
            "allow_partial_signals": int(allow_partial_signals),
            "require_macd": int(require_macd),
            "require_ma50": int(require_ma50),
            "require_rsi_gt": int(require_rsi_gt),
        }
        # evaluate across universe
        agg = evaluate_params_on_universe_with_mc(symbols, params, period=period, workers=workers, min_trades_required=min_trades_required)
        trial.set_user_attr("agg", agg)
        trial.set_user_attr("params", params)
        return float(agg.get("fitness", -10.0))

    return objective


# ---------------------------
# Runner
# ---------------------------
def run_optimize(symbols: List[str], n_trials: int = 100, period: str = "2y", workers: int = 4, study_name: str = None, min_trades_required: int = 3):

    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0, interval_steps=1)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner, study_name=None, storage=None, load_if_exists=True)
    objective = make_objective(symbols, period, workers, min_trades_required)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial
    best_agg = best.user_attrs.get("agg", {})
    best_params = best.user_attrs.get("params", best.params)

    out_json = "optuna_best_parameters_fixed_mc.json"
    with open(out_json, "w") as fh:
        json.dump({"fitness": best.value, "params": best_params, "agg": best_agg, "trial_number": best.number}, fh, indent=2)

    rows = []
    for t in study.trials:
        row = {"trial_number": t.number, "value": t.value}
        row.update(t.params)
        agg = t.user_attrs.get("agg", {})
        row.update({"trades": agg.get("trades", None), "sharpe_mean": agg.get("sharpe_mean", None), "total_return_mean": agg.get("total_return_mean", None), "max_drawdown_mean": agg.get("max_drawdown_mean", None), "mc_score": agg.get("mc_score", None)})
        rows.append(row)
    df = pd.DataFrame(rows)
    out_csv = "optuna_trials_fixed_mc.csv"
    df.to_csv(out_csv, index=False)

    print(f"[DONE] Best fitness: {best.value}, saved: {out_json}, trials: {out_csv}")
    return study, best, out_json, out_csv


# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    DEFAULT_MIN_TRADES = 3
    default_symbols = ["AAPL", "MSFT", "NVDA", "AMZN"]
    nasdaq_symbols = fetch_info()
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="+", default=nasdaq_symbols, help="Symbol list")
    p.add_argument("--trials", type=int, default=80, help="Number of Optuna trials")
    p.add_argument("--workers", type=int, default=8, help="Parallel workers for symbol evaluation (multiprocessing Pool size)")
    p.add_argument("--period", type=str, default="2y", help="yfinance history period")
    p.add_argument("--min_trades", type=int, default=DEFAULT_MIN_TRADES)
    args = p.parse_args()
    run_optimize(symbols=args.symbols, n_trials=args.trials, period=args.period, workers=args.workers, min_trades_required=args.min_trades)
