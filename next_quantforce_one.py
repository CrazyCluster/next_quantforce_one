#!/usr/bin/env python3
"""
next_quantforce_optimized.py

Optimized daily NASDAQ screener + weighted allocation + Alpaca Paper orderer.

Features:
- Batch yfinance download for many tickers
- Indicators via `ta` (ATR, RSI, MACD, MA20, MA50, ADX)
- Robust buy logic (Breakout + IRG + ADX + Volume + MA50 + MACD)
- Scoring and capital allocation (fractional or integer shares)
- Alpaca order placement with safety checks
- Logging and config-friendly
"""

import os
import time
import json
import logging
import requests
from typing import List, Dict
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import yfinance as yf
import ta

# Alpaca
try:
    import alpaca_trade_api as tradeapi
except Exception:
    tradeapi = None

# optional: load .env
from dotenv import load_dotenv
load_dotenv()

# -----------------------------------------------------------
# CONFIG
# -----------------------------------------------------------
# Alpaca config (set via env vars or replace here)
ALPACA_API_KEY = os.getenv("APCA_API_KEY_ID", "")
ALPACA_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY", "")
ALPACA_BASE_URL = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

# Screener params
DATA_PERIOD = "1y"             # history length for indicators
DATA_INTERVAL = "1d"
MIN_HISTORY = 80               # require at least this many rows
BATCH_SIZE = 80                # chunk size for yf.download tickers (avoid huge single call)
AUTO_ADJUST = True

# Strategy params
SL_MULT = 2.5
TP_MULT = 3.5
MOMENTUM_DAYS = 7
MOMENTUM_PCT = 0.02
VOLUME_FACTOR = 1.2
ADX_THRESHOLD = 25
IRG_THRESHOLD = 1.0
MIN_WEIGHT = 0.01

# Strategy params Exit
MAX_HOLDING_DAYS = 60

# Allocation / ordering
MINIMAL_CAPITAL_ACCOUNT = 500.00
MIN_CAPITAL_PER_TRADE = 50.0
MAX_ORDERS_ACCOUNT = 5     # limit number of new buys per run
ALLOW_FRACTIONAL = False      # if your broker supports fractional shares



# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("quantforce")

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


def load_optuna_parameters(path="optuna_best_parameters_mc.json"):
    """
    Lädt die besten Optuna-Parameter aus JSON und gibt sie als Dictionary zurück.
    Fällt automatisch auf DEFAULT_PARAMS zurück, wenn Datei fehlt/korrupt ist.
    """
    if not os.path.exists(path):
        print(f"[WARN] {path} nicht gefunden — verwende Default-Parameter.")
        return None

    try:
        with open(path, "r") as f:
            data = json.load(f)

        # Optuna speichert typischerweise:
        # { "best_params": { ... }, "best_value": ... }
        if "params" in data:
            return data["params"]
        else:
            return None

    except Exception as e:
        print(f"[WARN] Fehler beim Laden von {path}: {e}")
        print("[INFO] Verwende Default-Parameter.")
        return None



# -----------------------------------------------------------
# UTIL: normalize
# -----------------------------------------------------------
def normalize(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    min_v = s.min()
    max_v = s.max()
    if abs(max_v - min_v) < 1e-12:
        return pd.Series(0.0, index=s.index)
    return (s - min_v) / (max_v - min_v)


# -----------------------------------------------------------
# 1) Fetch tickers (user's function or provide list)
# -----------------------------------------------------------
def fetch_symbols() -> List[str]:
    """
    Replace this with your method to gather NASDAQ-100 symbols.
    Example: from NASDAQ_100.fetch_info() or a static list.
    """
    try:
        # try to import user's module if exists
        syms = fetch_info()
        if isinstance(syms, dict):
            # maybe it returns dict symbol->sector
            syms = list(syms.keys())
        return list(syms)
    except Exception:
        # fallback small list for safety/demo (replace with full NASDAQ-100)
        logger.warning("Could not import NASDAQ_100.fetch_info(); using fallback symbol list")
        return ["AAPL", "MSFT", "NVDA", "AMD", "AMZN", "TSLA", "GOOGL"]


# -----------------------------------------------------------
# 2) Batch download price data for many tickers
# -----------------------------------------------------------
def download_batch(tickers: List[str], period=DATA_PERIOD, interval=DATA_INTERVAL, auto_adjust=AUTO_ADJUST) -> Dict[str, pd.DataFrame]:
    all_data = {}
    # chunk to avoid too-large requests
    for i in range(0, len(tickers), BATCH_SIZE):
        chunk = tickers[i:i+BATCH_SIZE]
        logger.info(f"Downloading chunk {i // BATCH_SIZE + 1}: {len(chunk)} tickers")
        try:
            panel = yf.download(chunk, period=period, interval=interval, group_by="ticker", auto_adjust=auto_adjust, progress=False)
        except Exception as e:
            logger.exception("yf.download failed for chunk", exc_info=e)
            time.sleep(2)
            continue

        # if single ticker, then panel is normal df
        if isinstance(panel.columns, pd.MultiIndex):
            for t in chunk:
                if t in panel.columns.levels[0]:
                    df = panel[t].copy()
                    if df.empty:
                        continue
                    all_data[t] = df
                else:
                    # sometimes ticker not present
                    logger.debug(f"{t} not in downloaded panel")
        else:
            # single ticker requested
            # assign to first chunk ticker
            if len(chunk) == 1:
                all_data[chunk[0]] = panel.copy()
            else:
                # this case unexpected
                logger.warning("Unexpected panel shape from yf.download")
    return all_data


# -----------------------------------------------------------
# 3) Compute indicators using ta (handles missing safely)
# -----------------------------------------------------------
def compute_indicators_ta(df: pd.DataFrame) -> pd.DataFrame:
    """Compute indicators used by the tuneable strategy."""
    df = df.copy()
    if len(df) < 60:
        # Return DataFrame with empty indicator columns but no crash
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
            ATR_mean50=np.nan,
            rATR=np.nan,
            ATR50=np.nan
        )

    # --- ATR ---
    try:
        atr_indicator = ta.volatility.AverageTrueRange(
            high=df['High'], low=df['Low'], close=df['Close'], window=14
        )
        df['ATR'] = atr_indicator.average_true_range()
    except Exception:
        df['ATR'] = np.nan

    # --- RSI ---
    try:
        df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
    except Exception:
        df['RSI'] = np.nan

    # --- MACD ---
    try:
        macd = ta.trend.MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_hist'] = macd.macd_diff()
    except Exception:
        df[['MACD', 'MACD_signal', 'MACD_hist']] = np.nan

    # --- Moving Averages ---
    try:
        df['MA20'] = ta.trend.SMAIndicator(close=df['Close'], window=20).sma_indicator()
        df['MA50'] = ta.trend.SMAIndicator(close=df['Close'], window=50).sma_indicator()
    except Exception:
        df[['MA20', 'MA50']] = np.nan

    # --- ADX ---
    try:
        df['ADX'] = ta.trend.ADXIndicator(
            high=df['High'], low=df['Low'], close=df['Close'], window=14
        ).adx()
    except Exception:
        df['ADX'] = np.nan

    # --- Breakout (High 20) ---
    try:
        df['High_20'] = df['High'].rolling(window=20).max().shift(1)
    except Exception:
        df['High_20'] = np.nan

    # --- IRG indicator ---
    try:
        df['IRG'] = (df['Close'] - df['MA20']) / df['ATR']
    except Exception:
        df['IRG'] = np.nan

    # --- Volume regime ---
    try:
        df['Vol20'] = df['Volume'].rolling(20).mean()
    except Exception:
        df['Vol20'] = np.nan

    # --- ATR 50-day average ---
    try:
        df['ATR_mean50'] = df['ATR'].rolling(50).mean()
        df['ATR50'] = df['ATR_mean50']   # alias for readability
    except Exception:
        df['ATR_mean50'] = np.nan
        df['ATR50'] = np.nan

    # --- rATR: relative volatility (ATR / Close) ---
    try:
        df['rATR'] = df['ATR'] / df['Close'].replace(0, np.nan)
    except Exception:
        df['rATR'] = np.nan

    return df

def apply_strategy_tuneable(
    df: pd.DataFrame,
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
    allow_partial_signals: bool = False,  # NEW
    **kwargs
    ) -> pd.DataFrame:
    """Tunable buy/sell strategy with volatility-regime filtering and additional exits."""
    df = df.copy()

    needed = ['ATR', 'ATR50', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist',
              'MA20', 'MA50', 'ADX', 'High_20', 'IRG', 'Vol20']
    if not all(c in df.columns for c in needed):
        df = compute_indicators_ta(df)

    # --- NEW: relative ATR + Volatility regimes ---
    df['rATR'] = df['ATR'] / df['Close']

    df['Buy'] = False
    df['Sell'] = False
    df['Position'] = 0
    df['Entry_price'] = np.nan
    df['Entry_index'] = np.nan
    df['Stop_Loss'] = np.nan
    df['Take_Profit'] = np.nan
    df['Exit_Reason'] = None

    in_position = False
    entry_price = None
    entry_index = None
    current_SL = None
    current_TP = None

    start = max(60, momentum_days + 2)

    for i in range(start, len(df)):
        row = df.iloc[i]

        # -------- MOMENTUM --------
        if i >= momentum_days:
            past_close = df['Close'].iloc[i - momentum_days]
            momentum_pct = (row['Close'] - past_close) / past_close if past_close != 0 else 0.0
        else:
            momentum_pct = 0.0

        # -------- VOLUME --------
        avg_vol = df['Vol20'].iloc[i] if not pd.isna(df['Vol20'].iloc[i]) else 0.0
        vol_ok = (avg_vol > 0) and (row['Volume'] > volume_factor * avg_vol)

        # -------- BREAKOUT & SIGNAL FILTER --------
        breakout_ok = (not pd.isna(df['High_20'].iloc[i])) and (row['Close'] > df['High_20'].iloc[i])
        irg_ok = (not pd.isna(row['IRG'])) and (row['IRG'] >= irg_threshold)
        adx_ok = (not pd.isna(row['ADX'])) and (row['ADX'] >= adx_threshold)

        ma50_ok = (not require_ma50) or ((not pd.isna(row['MA50'])) and (row['Close'] > row['MA50']))
        macd_ok = (not require_macd) or (
            (not pd.isna(row['MACD'])) and (not pd.isna(row['MACD_signal'])) and (row['MACD'] > row['MACD_signal'])
        )
        rsi_ok = row.get('RSI', 0) > require_rsi_gt
        momentum_ok = momentum_pct >= min_momentum_pct

        # -------- VOLATILITY REGIME FILTER (NEW) --------
        rATR = row['rATR']

        vol_buy_ok = rATR < 0.030     # only low/medium

        # Volatility Spike filter
        atr_spike = (row['ATR'] > row['ATR50'] * 1.6)

        # BUY condition
        if allow_partial_signals:
            checks = [breakout_ok, ma50_ok, adx_ok, macd_ok,
                      irg_ok, momentum_ok, vol_ok, rsi_ok, vol_buy_ok]
            buy_cond = sum(bool(x) for x in checks) >= 5
        else:
            buy_cond = (
                breakout_ok and ma50_ok and adx_ok and macd_ok and
                irg_ok and momentum_ok and vol_ok and rsi_ok and vol_buy_ok and
                (not atr_spike)
            )

        # -------- BUY --------
        if (not in_position) and buy_cond:
            in_position = True
            entry_price = row['Close']
            entry_index = i

            atr = row['ATR'] if not pd.isna(row['ATR']) else 0.0
            current_SL = entry_price - sl_mult * atr
            current_TP = entry_price + tp_mult * atr

            df.at[df.index[i], 'Buy'] = True
            df.at[df.index[i], 'Entry_price'] = entry_price
            df.at[df.index[i], 'Entry_index'] = i
            df.at[df.index[i], 'Stop_Loss'] = current_SL
            df.at[df.index[i], 'Take_Profit'] = current_TP
            df.at[df.index[i], 'Position'] = 1
            continue

        # -------- SELL --------
        if in_position:
            price = row['Close']
            exit_reason = None

            # 1. SL / TP
            if price <= current_SL:
                exit_reason = 'SL'
            elif price >= current_TP:
                exit_reason = 'TP'

            # 2. NEW: Volatility Spike Exit (>35%)
            elif row['ATR'] > row['ATR50'] * 1.35:
                exit_reason = 'ATR_spike_exit'

            # 3. Trend Break (Close < MA20 + MA20 falling)
            elif (not pd.isna(row['MA20'])) and price < row['MA20']:
                prev_ma20 = df['MA20'].iloc[i - 1]
                if prev_ma20 is not None and row['MA20'] < prev_ma20:
                    exit_reason = 'Trend_break'

            # 4. MACD loss (if enabled)
            elif (not pd.isna(row['MACD'])) and (not pd.isna(row['MACD_signal'])) and (row['MACD'] < row['MACD_signal']):
                exit_reason = 'MACD_loss'

            # 5. ADX collapse
            elif (not pd.isna(row['ADX'])) and (row['ADX'] < max(12, adx_threshold * 0.6)):
                exit_reason = 'ADX_fall'

            # 6. NEW: Time-Based Exit (if enabled)
            if exit_reason is None and MAX_HOLDING_DAYS is not None:
                if (i - entry_index) >= MAX_HOLDING_DAYS:
                    exit_reason = 'TBE'

            if exit_reason:
                df.at[df.index[i], 'Sell'] = True
                df.at[df.index[i], 'Exit_Reason'] = exit_reason
                df.at[df.index[i], 'Position'] = 0

                in_position = False
                entry_price = None
                entry_index = None
                current_SL = None
                current_TP = None
            else:
                df.at[df.index[i], 'Position'] = 1

        else:
            df.at[df.index[i], 'Position'] = 0

    return df


# -----------------------------------------------------------
# 4) Generate buy/sell signals for a single ticker df
# -----------------------------------------------------------
def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = compute_indicators_ta(df)

    # Prepare result rows: we'll only keep rows where Buy or Sell happens
    df["Buy"] = False
    df["Sell"] = False
    df["Entry_price"] = np.nan
    df["Stop_Loss"] = np.nan
    df["Take_Profit"] = np.nan

    in_position = False
    entry_price = None
    current_SL = None
    current_TP = None

    # start index after indicators have warmed up
    start = max(MIN_HISTORY, 60)
    for i in range(start, len(df)):
        row = df.iloc[i]
        # momentum
        if i >= MOMENTUM_DAYS:
            past_close = df["Close"].iloc[i - MOMENTUM_DAYS]
            momentum_pct = (row["Close"] - past_close) / past_close
        else:
            momentum_pct = 0.0

        # checks
        avg_vol = df["Vol20"].iloc[i] if not pd.isna(df["Vol20"].iloc[i]) else 0.0
        vol_ok = (avg_vol > 0) and (row["Volume"] > VOLUME_FACTOR * avg_vol)
        breakout_ok = (not pd.isna(df["High_20"].iloc[i])) and (row["Close"] > df["High_20"].iloc[i])
        irg_ok = (not pd.isna(row["IRG"])) and (row["IRG"] >= IRG_THRESHOLD)
        adx_ok = (not pd.isna(row["ADX"])) and (row["ADX"] >= ADX_THRESHOLD)
        ma50_ok = (not pd.isna(row["MA50"])) and (row["Close"] > row["MA50"])
        macd_ok = (not pd.isna(row["MACD"])) and (not pd.isna(row["MACD_signal"])) and (row["MACD"] > row["MACD_signal"])
        momentum_ok = momentum_pct >= MOMENTUM_PCT

        buy_cond = breakout_ok and ma50_ok and adx_ok and macd_ok and irg_ok and momentum_ok and vol_ok and (row["RSI"] > 30)

        if (not in_position) and buy_cond:
            in_position = True
            entry_price = row["Close"]
            atr = row["ATR"] if not pd.isna(row["ATR"]) else 0.0
            current_SL = entry_price - SL_MULT * atr
            current_TP = entry_price + TP_MULT * atr

            df.at[df.index[i], "Buy"] = True
            df.at[df.index[i], "Entry_price"] = entry_price
            df.at[df.index[i], "Stop_Loss"] = current_SL
            df.at[df.index[i], "Take_Profit"] = current_TP
            continue

        if in_position:
            price = row["Close"]
            exit_reason = None
            if (not pd.isna(current_SL)) and (price <= current_SL):
                exit_reason = "SL"
            elif (not pd.isna(current_TP)) and (price >= current_TP):
                exit_reason = "TP"
            elif (not pd.isna(row["MA20"])) and (price < row["MA20"]):
                exit_reason = "MA20_break"
            elif (not pd.isna(row["MACD"])) and (not pd.isna(row["MACD_signal"])) and (row["MACD"] < row["MACD_signal"]):
                exit_reason = "MACD_loss"
            elif (not pd.isna(row["ADX"])) and (row["ADX"] < 20):
                exit_reason = "ADX_fall"

            if exit_reason:
                df.at[df.index[i], "Sell"] = True
                # reset
                in_position = False
                entry_price = None
                current_SL = None
                current_TP = None

    # return rows that contain signals
    signals = df[(df["Buy"] == True) | (df["Sell"] == True)].copy()
    return signals


# -----------------------------------------------------------
# 5) Compute strength score for Buy rows across all symbols
# -----------------------------------------------------------
def compute_strength_score(signals_df: pd.DataFrame) -> pd.DataFrame:
    if signals_df is None or signals_df.empty:
        return pd.DataFrame()

    df = signals_df.copy().reset_index(drop=False)  # keep original index in 'index' column

    buys = df[df["Buy"] == True].copy()
    if buys.empty:
        return pd.DataFrame()

    # momentum 7 days (approx): use pct change on Close grouped by symbol
    buys["Momentum7"] = buys.groupby("Symbol")["Close"].transform(lambda x: x.pct_change(periods=7)).fillna(0.0)

    # components normalized
    irg_n = normalize(buys.get("IRG", pd.Series(0.0, index=buys.index)))
    adx_n = normalize(buys.get("ADX", pd.Series(0.0, index=buys.index)))
    macd_n = normalize(buys.get("MACD_hist", pd.Series(0.0, index=buys.index)))
    mom_n = normalize(buys["Momentum7"])
    vol_n = normalize((buys.get("Volume", 0.0) / buys.get("Vol20", 1.0)).replace([np.inf, -np.inf], 0.0).fillna(0.0))

    buys["Strength_Score"] = (0.30 * irg_n + 0.20 * adx_n + 0.20 * macd_n + 0.20 * mom_n + 0.10 * vol_n).clip(lower=0.0)

    return buys


# -----------------------------------------------------------
# 6) Capital allocation
# -----------------------------------------------------------
def allocate_capital(buys_df: pd.DataFrame, total_capital: float = 0.00,
                     min_weight: float = MIN_WEIGHT, allow_fractional: bool = ALLOW_FRACTIONAL,
                     min_capital_per_trade: float = MIN_CAPITAL_PER_TRADE, debug: bool = False) -> pd.DataFrame:
    if buys_df is None or buys_df.empty:
        return pd.DataFrame()

    df = buys_df.copy()
    # ensure Strength_Score present
    if "Strength_Score" not in df.columns:
        df = compute_strength_score(df)
        if df.empty:
            return pd.DataFrame()

    # pick latest buy per symbol
    df_latest = df.sort_values("index").groupby("Symbol").tail(1).set_index("Symbol")

    # ensure numeric price
    df_latest["Close"] = pd.to_numeric(df_latest.get("Close", pd.Series(np.nan, index=df_latest.index)), errors="coerce")
    df_latest["Strength_Score"] = pd.to_numeric(df_latest.get("Strength_Score", 0.0), errors="coerce").fillna(0.0)

    scores = df_latest["Strength_Score"].astype(float)
    total_score = scores.sum()

    if total_score <= 0:
        # fallback equal weighting
        weights = pd.Series(1.0 / len(scores), index=scores.index)
    else:
        weights = scores / total_score

    weights = weights.clip(lower=min_weight)
    weights = weights / weights.sum()

    capital_alloc = weights * total_capital
    allocation = pd.DataFrame({
        "Date": df_latest["Date"],
        "Strength_Score": scores,
        "Weight": weights,
        "Capital": capital_alloc,
        "Price": df_latest["Close"],
        "Take_Profit": df_latest["Take_Profit"].fillna(0).astype(float),
        "Stop_Loss": df_latest["Stop_Loss"].fillna(0).astype(float)
    })

    # drop nan prices
    allocation = allocation[allocation["Price"].notna()].copy()
    if allocation.empty:
        if debug:
            logger.warning("No valid prices for allocation")
        return pd.DataFrame()

    if allow_fractional:
        allocation["Shares"] = allocation["Capital"] / allocation["Price"]
        allocation.loc[allocation["Capital"] < min_capital_per_trade, "Shares"] = 0.0
    else:
        allocation["Shares"] = (allocation["Capital"] / allocation["Price"]).fillna(0).astype(int)
        allocation.loc[allocation["Capital"] < min_capital_per_trade, "Shares"] = 0

    allocation["Allocated_Capital"] = allocation["Shares"] * allocation["Price"]

    # fallback: if all shares 0 and fractional not allowed, try buy cheapest that fits total capital
    if allocation["Shares"].sum() == 0 and not allow_fractional:
        if debug:
            logger.info("All integer shares are zero; trying cheapest fallback")
        cheapest = allocation.sort_values("Price").iloc[0]
        if cheapest["Price"] <= total_capital:
            sym = allocation.sort_values("Price").index[0]
            allocation.loc[sym, "Shares"] = int(total_capital // allocation.loc[sym, "Price"])
            allocation["Allocated_Capital"] = allocation["Shares"] * allocation["Price"]
            allocation["Capital"] = allocation["Allocated_Capital"]
        else:
            if debug:
                logger.info("Fallback impossible without fractional shares.")

    return allocation.reset_index()

# ---------------------------
# Alpaca client wrapper
# ---------------------------
class AlpacaClient:
    def __init__(self, api_key: str, secret_key: str, base_url: str):
        if tradeapi is None:
            raise RuntimeError("alpaca-trade-api not installed")
        self.api = tradeapi.REST(api_key, secret_key, base_url=base_url, api_version='v2')


    def get_account_cash(self):
        try:
            account = self.api.get_account()
            return round(float(account.cash), 2)
        except Exception:
            logger.exception("Failed to fetch positions")
            return 0.00

    def get_positions(self) -> Dict[str, float]:
        try:
            positions = self.api.list_positions()
            return {p.symbol: float(p.qty) for p in positions}
        except Exception:
            logger.exception("Failed to fetch positions")
            return {}

    def get_open_orders(self):
        try:
            return self.api.list_orders(status="open")
        except Exception:
            return []

    def submit_buy(self, symbol: str, qty: float, side="buy", time_in_force="day", type="market"):
        try:
            if qty <= 0:
                logger.debug(f"Not placing order for {symbol} qty {qty}")
                return None
            if isinstance(qty, float) and not ALLOW_FRACTIONAL:
                qty = int(qty)
            order = self.api.submit_order(symbol=symbol, qty=qty, side=side, type=type, time_in_force=time_in_force)
            logger.info(f"Placed order {order.id} {symbol} {qty}")
            return order
        except Exception:
            logger.exception(f"Order failed for {symbol} qty {qty}")
            return None

    def submit_buy_tp_sl(self, symbol: str, qty: float, tp_price: float, sl_price: float, side="buy", time_in_force="day", type="market", order_class="bracket"):
        try:
            if qty <= 0:
                logger.debug(f"Not placing order for {symbol} qty {qty}")
                return None
            if isinstance(qty, float) and not ALLOW_FRACTIONAL:
                qty = int(qty)
            order = self.api.submit_order(symbol=symbol, qty=qty, side=side, type=type, time_in_force=time_in_force, order_class=order_class, take_profit={"limit_price": tp_price}, stop_loss={'stop_price': sl_price})
            logger.info(f"Placed order {order.id} {symbol} {qty}")
            return order
        except Exception:
            logger.exception(f"Order failed for {symbol} qty {qty}")
            return None

    def submit_sell(self, symbol: str, qty: float, side="sell", time_in_force="day", type="market"):
        try:
            if qty <= 0:
                logger.debug(f"Not placing sell for {symbol} qty {qty}")
                return None
            if isinstance(qty, float) and not ALLOW_FRACTIONAL:
                qty = int(qty)
            order = self.api.submit_order(symbol=symbol, qty=qty, side=side, type=type, time_in_force=time_in_force)
            logger.info(f"Placed SELL order {order.id} {symbol} {qty}")
            return order
        except Exception:
            logger.exception(f"Sell failed for {symbol} qty {qty}")
            return None

# ---------------------------
# Main runner with Buy + Sell
# ---------------------------

def run_daily(load_paramaters=True):
    paramters = None
    symbols = fetch_symbols()
    logger.info(f"Loaded {len(symbols)} symbols")
    data_map = download_batch(symbols, period=DATA_PERIOD, interval=DATA_INTERVAL, auto_adjust=AUTO_ADJUST)
    logger.info(f"Downloaded data for {len(data_map)} symbols")
    if load_paramaters:
        paramters = load_optuna_parameters("optuna_best_parameters_mc.json")
    all_signals = []
    for sym, df in data_map.items():
        try:
            if paramters:
                df = apply_strategy_tuneable(df, **paramters)
                signals = df[(df["Buy"] == True) | (df["Sell"] == True)].copy()
            else:
                signals = generate_signals(df)
            if not signals.empty:
                signals["Symbol"] = sym
                all_signals.append(signals)
        except Exception:
            logger.exception(f"Signal gen failed for {sym}")

    if not all_signals:
        logger.info("No signals across universe")
        return None

    # Use only last signal of symbols
    signals_df = pd.concat(all_signals).sort_index()
    latest_per_symbol = signals_df.groupby("Symbol").tail(1).reset_index()
    # Use only todays signals
    latest_date = latest_per_symbol["Date"].max()
    latest_per_symbol = latest_per_symbol[latest_per_symbol["Date"] == latest_date]
    # SELL side: find held positions that have Sell signal
    sell_candidates = latest_per_symbol[latest_per_symbol["Sell"] == True].copy()
    #logger.info("-----Selling Candidates-----")
    #display_cols = ["Date", "Symbol", "Close", "Exit_Reason"]
    #logger.info(sell_candidates[display_cols].to_string(index=False))
    # BUY side: latest rows that are Buy
    candidate_buys = latest_per_symbol[latest_per_symbol["Buy"] == True].copy()
    #logger.info("-----Buying Candidates-----")
    #display_cols = ["Date", "Symbol", "Close"]
    #logger.info(candidate_buys[display_cols].to_string(index=False))

    # Real trading: require Alpaca keys
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        logger.error("Alpaca API keys not set; cannot place orders.")
        return None

    client = AlpacaClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL)

    # SELL execution first: close positions for sell_signals
    positions = client.get_positions()
    open_orders = client.get_open_orders()

    if not sell_candidates.empty:
        logger.info(f"Found {len(sell_candidates)} sell signals to evaluate")
        for row in sell_candidates.itertuples(index=False):
            sym = row.Symbol
            if sym not in positions:
                logger.info(f"Sell signal for {sym} but not held; skipping")
                continue
            qty = positions[sym]
            # skip if there's already an open order for the symbol
            if any(getattr(o, 'symbol', None) == sym for o in open_orders):
                logger.info(f"Open order exists for {sym}; skipping sell")
                continue
            logger.info(f"Placing SELL market for {sym} qty {qty}")
            client.submit_sell(sym, qty)
            time.sleep(0.25)
    else:
        logger.info("No sell signals to execute.")

    account_cash = client.get_account_cash()
    positions = client.get_positions()

    # Score and allocation for buys
    if candidate_buys.empty:
        logger.info("No candidate buys today")
    elif (account_cash - MINIMAL_CAPITAL_ACCOUNT) <= 0:
        logger.info("No cash for buys today")
    else:
        scored = compute_strength_score(candidate_buys)
        if not scored.empty:
            scored_sorted = scored.sort_values("Strength_Score", ascending=False)
            no_orders = min(MAX_ORDERS_ACCOUNT - len(positions), len(scored_sorted))
            top = scored_sorted.head(no_orders)
            allocation = allocate_capital(top, total_capital=account_cash, min_weight=MIN_WEIGHT, allow_fractional=ALLOW_FRACTIONAL, min_capital_per_trade=MIN_CAPITAL_PER_TRADE, debug=True)
            if allocation.empty:
                logger.info("Allocation returned empty")
            else:
                logger.info("Planned allocation:")
                display_cols = ["Date", "Shares", "Symbol", "Price", "Weight", "Capital"]
                logger.info(allocation[display_cols].to_string(index=False))
        else:
            allocation = pd.DataFrame()
            logger.info("No scored buys")


# BUY execution: place new buys for allocation if not already owned and no open orders
    positions = client.get_positions()
    open_orders = client.get_open_orders()
    orders_placed = []
    if 'allocation' in locals() and not allocation.empty:
        for row in allocation.itertuples(index=False):
            sym = row.Symbol if hasattr(row, 'Symbol') else row[0]
            shares = float(row.Shares)
            if sym in positions:
                logger.info(f"Already own {sym}, skipping buy")
                continue
            if any(getattr(o, 'symbol', None) == sym for o in open_orders):
                logger.info(f"Open order exists for {sym}, skipping buy")
                continue
            logger.info(f"Placing BUY market for {sym} qty {shares}")
            tp_price  = getattr(row, "Take_Profit", False)
            sl_price = getattr(row, "Stop_Loss", False)
            if all([tp_price, sl_price]):
                order = client.submit_buy_tp_sl(sym, shares, round(tp_price, 2), round(sl_price, 2))
            else:
                order = client.submit_buy(sym, shares)
            if order:
                orders_placed.append((sym, shares, getattr(order, 'id', None)))
            time.sleep(0.25)

    logger.info(f"Orders placed: {orders_placed}")
    return {"buys": orders_placed, "sells_executed": len(sell_candidates)}

#----------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    res = run_daily(load_paramaters=True)
    logger.info(res)
