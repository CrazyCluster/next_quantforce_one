import pandas as pd
import numpy as np
import yfinance as yf
from itertools import product
from NASDAQ_100 import fetch_info
import warnings, ta
warnings.filterwarnings("ignore")


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

# ----------------------------------------
# Utility: Evaluate trades from strategy
# ----------------------------------------
def evaluate_performance(df):
    df = df.copy()
    df["Return"] = 0.0
    returns = []

    position_open = False
    entry_price = None

    for i in range(len(df)):
        if df["Buy"].iloc[i] and not position_open:
            entry_price = df["Close"].iloc[i]
            position_open = True

        if df["Sell"].iloc[i] and position_open:
            exit_price = df["Close"].iloc[i]
            returns.append((exit_price - entry_price) / entry_price)
            position_open = False

    if returns:
        total_return = np.prod([1+r for r in returns]) - 1
        max_dd = max_drawdown(returns)
        fitness = total_return / max(0.01, max_dd)
        return total_return, max_dd, fitness

    return 0, 0, 0


# ----------------------------------------
# Max Drawdown
# ----------------------------------------
def max_drawdown(returns):
    equity = np.cumprod([1+r for r in returns])
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return abs(dd.min())


# ----------------------------------------
# Single-symbol backtest
# ----------------------------------------
def run_backtest(symbol, params, period="1y"):
    try:
        df = yf.download(symbol, period=period, auto_adjust=True, progress=False)
        if df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            try:
                # choose 'Close','High','Low','Volume' - if nested, take level 1
                first = df.columns.levels[1][0]
                df = df.xs(first, axis=1, level=1)
            except Exception:
                df.columns = [str(c) for c in df.columns]

        df = compute_indicators_ta(df)
        df = apply_strategy(df, **params)

        return evaluate_performance(df)

    except Exception:
        return None


# ----------------------------------------
# Grid search optimization
# ----------------------------------------
def optimize_parameters(symbols):
    # Parameter ranges
    param_grid = {
        "adx_threshold": [20, 25, 30],
        "irg_threshold": [0.8, 1.0, 1.2],
        "min_momentum_pct": [0.015, 0.02, 0.03],
        "volume_factor": [1.1, 1.2, 1.3],
        "sl_mult": [2.0, 2.5, 3.0],
        "tp_mult": [3.0, 3.5, 4.0],
    }

    keys = list(param_grid.keys())
    best = {"fitness": -999, "params": None}

    for combo in product(*param_grid.values()):
        params = dict(zip(keys, combo))
        fitness_scores = []

        print("Test:", params)

        for symbol in symbols:
            result = run_backtest(symbol, params)
            if result:
                _, _, fitness = result
                fitness_scores.append(fitness)

        if fitness_scores:
            avg_fitness = np.mean(fitness_scores)
            if avg_fitness > best["fitness"]:
                best = {"fitness": avg_fitness, "params": params}
                print("🔻 Neuer Bestwert:", best)

    return best

def run_test():
    sym = 'AAPL'
    print('Downloading demo data for:', sym)
    df = yf.download(sym, period='2y', auto_adjust=True, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
            try:
                # choose 'Close','High','Low','Volume' - if nested, take level 1
                first = df.columns.levels[1][0]
                df = df.xs(first, axis=1, level=1)
            except Exception:
                df.columns = [str(c) for c in df.columns]

    df = compute_indicators_ta(df)
    res = apply_strategy(df, allow_partial_signals=True)
    # print(res[(res['Buy'] | res['Sell'])].tail(20).to_string())
    print(res[["Buy", "Sell"]].sum())

#----------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    # run_test()
    symbols = ["AAPL", "MSFT", "NVDA", "AMZN", "META"]
    nasdaq_symbols = fetch_info()   # oder NASDAQ100
    best_params = optimize_parameters(symbols)
    print("🎉 Beste Parameter:", best_params)
