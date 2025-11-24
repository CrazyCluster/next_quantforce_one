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

# Allocation / ordering
TOTAL_CAPITAL = 10000.0      # capital to allocate (paper account)
ALLOW_FRACTIONAL = True      # if your broker supports fractional shares
MIN_CAPITAL_PER_TRADE = 50.0
MAX_ORDERS_PER_DAY = 10      # limit number of new buys per run

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



def load_optuna_parameters(path="optuna_best_parameters.json"):
    """
    Lädt die besten Optuna-Parameter aus JSON und gibt sie als Dictionary zurück.
    Fällt automatisch auf DEFAULT_PARAMS zurück, wenn Datei fehlt/korrupt ist.
    """
    if not os.path.exists(path):
        print(f"[WARN] {path} nicht gefunden — verwende Default-Parameter.")
        return DEFAULT_PARAMS

    try:
        with open(path, "r") as f:
            data = json.load(f)

        # Optuna speichert typischerweise:
        # { "best_params": { ... }, "best_value": ... }
        if "params" in data:
            return data["params"]


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
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if len(df) < MIN_HISTORY:
        # create columns with NaNs so later code can handle gracefully
        for col in ["ATR", "RSI", "MACD", "MACD_signal", "MACD_hist", "MA20", "MA50", "ADX", "High_20", "IRG", "Vol20"]:
            df[col] = np.nan
        return df

    # ATR
    try:
        df["ATR"] = ta.volatility.AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"], window=14).average_true_range()
    except Exception:
        df["ATR"] = np.nan

    # RSI
    try:
        df["RSI"] = ta.momentum.RSIIndicator(close=df["Close"], window=14).rsi()
    except Exception:
        df["RSI"] = np.nan

    # MACD
    try:
        macd = ta.trend.MACD(close=df["Close"], window_slow=26, window_fast=12, window_sign=9)
        df["MACD"] = macd.macd()
        df["MACD_signal"] = macd.macd_signal()
        df["MACD_hist"] = macd.macd_diff()
    except Exception:
        df["MACD"] = df["MACD_signal"] = df["MACD_hist"] = np.nan

    # Moving averages
    try:
        df["MA20"] = ta.trend.SMAIndicator(close=df["Close"], window=20).sma_indicator()
        df["MA50"] = ta.trend.SMAIndicator(close=df["Close"], window=50).sma_indicator()
    except Exception:
        df["MA20"] = df["MA50"] = np.nan

    # ADX
    try:
        df["ADX"] = ta.trend.ADXIndicator(high=df["High"], low=df["Low"], close=df["Close"], window=14).adx()
    except Exception:
        df["ADX"] = np.nan

    # High 20 (shifted by 1)
    df["High_20"] = df["High"].rolling(window=20).max().shift(1)

    # IRG
    df["IRG"] = (df["Close"] - df["MA20"]) / df["ATR"]

    # Vol20
    df["Vol20"] = df["Volume"].rolling(20).mean()

    return df


# -----------------------------------------------------------
# 4) Generate buy/sell signals for a single ticker df
# -----------------------------------------------------------
def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = compute_indicators(df)

    data = load_optuna_parameters()

    if data:
        MOMENTUM_DAYS = data["momentum_days"]
        VOLUME_FACTOR = data["volume_factor"]
        IRG_THRESHOLD = data["irg_threshold"]
        ADX_THRESHOLD = data["adx_threshold"]
        MOMENTUM_PCT = data["min_momentum_pct"]
        SL_MULT = data["sl_mult"]
        TP_MULT = data["tp_mult"]

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
def allocate_capital(buys_df: pd.DataFrame, total_capital: float = TOTAL_CAPITAL,
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
        "Strength_Score": scores,
        "Weight": weights,
        "Capital": capital_alloc,
        "Price": df_latest["Close"]
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

def run_daily(cash_to_allocate: float = TOTAL_CAPITAL):
    symbols = fetch_symbols()
    logger.info(f"Loaded {len(symbols)} symbols")

    data_map = download_batch(symbols, period=DATA_PERIOD, interval=DATA_INTERVAL, auto_adjust=AUTO_ADJUST)
    logger.info(f"Downloaded data for {len(data_map)} symbols")

    all_signals = []
    for sym, df in data_map.items():
        try:
            signals = generate_signals(df)
            if not signals.empty:
                signals["Symbol"] = sym
                all_signals.append(signals)
        except Exception:
            logger.exception(f"Signal gen failed for {sym}")

    if not all_signals:
        logger.info("No signals across universe")
        return None

    signals_df = pd.concat(all_signals).sort_index()

    latest_per_symbol = signals_df.groupby("Symbol").tail(1).reset_index()

    # SELL side: find held positions that have Sell signal
    sell_candidates = latest_per_symbol[latest_per_symbol["Sell"] == True].copy()

    # BUY side: latest rows that are Buy
    candidate_buys = latest_per_symbol[latest_per_symbol["Buy"] == True].copy()

    # Score and allocation for buys
    if candidate_buys.empty:
        logger.info("No candidate buys today")
    else:
        scored = compute_strength_score(candidate_buys)
        if not scored.empty:
            scored_sorted = scored.sort_values("Strength_Score", ascending=False)
            top = scored_sorted.head(MAX_ORDERS_PER_DAY)
            allocation = allocate_capital(top, total_capital=cash_to_allocate, min_weight=MIN_WEIGHT, allow_fractional=ALLOW_FRACTIONAL, min_capital_per_trade=MIN_CAPITAL_PER_TRADE, debug=True)
            if allocation.empty:
                logger.info("Allocation returned empty")
            else:
                logger.info("Planned allocation:")
                logger.info(allocation.to_string(index=False))
        else:
            allocation = pd.DataFrame()
            logger.info("No scored buys")

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
    res = run_daily(cash_to_allocate=TOTAL_CAPITAL)
    if res is None:
        print("No actions")
    else:
        if isinstance(res, dict):
            if 'allocation' in res:
                print("Planned buys:")
                print(res['allocation'].to_string(index=False))
            if 'sells' in res:
                print("Planned sells (latest sell signals):")
                print(res['sells'].to_string(index=False))
            else:
                print(res)
