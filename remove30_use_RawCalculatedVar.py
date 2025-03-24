#!/usr/bin/env python3
"""
Mechanical HFT Agent using Raw Variance-Based SL/TP (No 30-Point Floor)

Features:
1. Continuously fetches tick data from MT5 for a given symbol, storing all ticks in raw_tick_data.csv.
2. Maintains a 30-tick rolling buffer for slope & variance calculations.
3. Every 30 seconds, if slope > threshold => BUY; if slope < -threshold => SELL.
4. SL/TP are set to +/- the raw computed variance (no forced 30-point floor).
5. Aggregated data (prices, variance, slope, trend) is logged to tick_data.csv.
6. Each order request is logged to order_data.csv.
7. Avoids UnboundLocalError by properly initializing latest_tick_time in main().

Author: [Your Name]
Pretend: A professional programmer with 10+ years of experience.
"""

import os
import MetaTrader5 as mt5
import time
import datetime
import numpy as np
from collections import deque
import logging
import csv
from scipy.stats import linregress

# -------------------------------
# Logging Configuration
# -------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -------------------------------
# Global Configuration
# -------------------------------
SYMBOL = "XAUUSD.sml"         # Adjust if broker uses "XAUUSD" or "XAUUSD.pro"
TICK_BUFFER_SIZE = 500         # Rolling buffer size
TICKS_PER_CALL = 200          # Max ticks to fetch
LOOKBACK_MINUTES = 2          # Start 2 minutes in the past
LOOP_ITERATIONS = 100         # We'll do up to 100 order attempts
SLEEP_BETWEEN_ORDERS = 30     # Seconds between each iteration

SLOPE_THRESHOLD = 0.0001      # Slope threshold for bullish/bearish signals
LOT_SIZE = 0.01               # Trading volume

# CSV Filenames
RAW_TICK_CSV = "raw_tick_data.csv"
TICK_CSV = "tick_data.csv"
ORDER_CSV = "order_data.csv"

# -------------------------------
# CSV: Raw Tick Data
# -------------------------------
def prepare_raw_tick_csv():
    """Create raw_tick_data.csv if it doesn't exist, with header: date, time, symbol, price"""
    if not os.path.exists(RAW_TICK_CSV):
        with open(RAW_TICK_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["date", "time", "symbol", "price"])

def append_raw_tick_csv_row(date_str, time_str, symbol, price):
    """Append a row to raw_tick_data.csv"""
    with open(RAW_TICK_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([date_str, time_str, symbol, f"{price:.5f}"])

# -------------------------------
# CSV: Aggregated Tick Data
# -------------------------------
def prepare_tick_csv():
    """Create tick_data.csv if it doesn't exist, with columns for aggregated stats."""
    if not os.path.exists(TICK_CSV):
        with open(TICK_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["date", "time", "symbol", "prices", "variance", "slope", "trend"])

def append_tick_csv_row(date_str, time_str, symbol, prices_str, variance, slope, trend):
    """Append a row of aggregated data to tick_data.csv"""
    with open(TICK_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([date_str, time_str, symbol, prices_str, variance, slope, trend])

# -------------------------------
# CSV: Order Data
# -------------------------------
def prepare_order_csv():
    """Create order_data.csv if it doesn't exist."""
    if not os.path.exists(ORDER_CSV):
        with open(ORDER_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Ticket", "Order", "Time", "Symbol", "Type", "Volume", "Price", "Profit", "Comment"])

def append_order_csv_row(ticket, order_num, time_str, symbol, order_type, volume, price, profit, comment):
    """Append a row to order_data.csv"""
    with open(ORDER_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            ticket,
            order_num,
            time_str,
            symbol,
            order_type,
            volume,
            f"{price:.5f}",
            f"{profit:.2f}",
            comment
        ])

# -------------------------------
# Initialize CSVs
# -------------------------------
prepare_raw_tick_csv()
prepare_tick_csv()
prepare_order_csv()

# -------------------------------
# Initialize MetaTrader 5
# -------------------------------
if not mt5.initialize():
    logging.error("MT5 initialization failed, error code=%s", mt5.last_error())
    quit()

# Ensure the symbol is recognized
if not mt5.symbol_select(SYMBOL, True):
    logging.error("symbol_select(%s) failed. Check symbol name or broker availability.", SYMBOL)
    mt5.shutdown()
    quit()

# Check if auto trading is enabled
terminal_info = mt5.terminal_info()
if terminal_info is None or not terminal_info.trade_allowed:
    logging.warning("AutoTrading is disabled in your MT5 terminal. Please enable it for order execution.")

# Rolling buffer for price data
tick_buffer = deque(maxlen=TICK_BUFFER_SIZE)

# -------------------------------
# Helper: fetch_new_ticks
# -------------------------------
def fetch_new_ticks(symbol, from_time):
    """Fetch up to TICKS_PER_CALL new ticks from 'from_time' onward."""
    ticks = mt5.copy_ticks_from(symbol, from_time, TICKS_PER_CALL, mt5.COPY_TICKS_ALL)
    return ticks

# -------------------------------
# place_market_order
# -------------------------------
def place_market_order(symbol, volume, order_type, price, variance):
    """
    Places a market order using raw variance for SL/TP (no forced 30-point floor).
    For BUY: TP = price + variance, SL = price - variance
    For SELL: TP = price - variance, SL = price + variance
    """
    if order_type == "BUY":
        tp = price + variance
        sl = price - variance
        mt5_order_type = mt5.ORDER_TYPE_BUY
        comment = f"Slope-based Buy (var={variance:.2f})"
    else:  # SELL
        tp = price - variance
        sl = price + variance
        mt5_order_type = mt5.ORDER_TYPE_SELL
        comment = f"Slope-based Sell (var={variance:.2f})"

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": mt5_order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,  # or ORDER_FILLING_IOC if broker supports partial fill
        "comment": comment
    }

    result = mt5.order_send(request)
    return result

# -------------------------------
# MAIN
# -------------------------------
def main():
    # Define latest_tick_time within main to avoid UnboundLocalError
    latest_tick_time = datetime.datetime.now() - datetime.timedelta(minutes=LOOKBACK_MINUTES)
    logging.info("Starting mechanical HFT agent for %s", SYMBOL)
    logging.info("Will attempt up to %d trades, every %d seconds.", LOOP_ITERATIONS, SLEEP_BETWEEN_ORDERS)
    logging.info("Initial start time: %s", latest_tick_time.strftime("%Y-%m-%d %H:%M:%S"))

    for order_num in range(1, LOOP_ITERATIONS + 1):
        # 1) Fetch new ticks
        ticks = fetch_new_ticks(SYMBOL, latest_tick_time)
        if ticks is not None and len(ticks) > 0:
            for tick in ticks:
                tick_dt = datetime.datetime.fromtimestamp(tick["time"])
                # Update latest_tick_time for next fetch
                if tick_dt > latest_tick_time:
                    latest_tick_time = tick_dt + datetime.timedelta(microseconds=1)

                price = tick["last"]
                if price <= 0:
                    price = (tick["bid"] + tick["ask"]) / 2

                # Append each raw tick
                date_str = tick_dt.strftime("%Y-%m-%d")
                time_str = tick_dt.strftime("%H:%M:%S.%f")[:-3]
                append_raw_tick_csv_row(date_str, time_str, SYMBOL, price)

                # Add to rolling buffer
                tick_buffer.append(price)
        else:
            logging.info("No new ticks or error: %s", mt5.last_error())

        # 2) If buffer is full, compute slope & variance every 30 sec
        if len(tick_buffer) == TICK_BUFFER_SIZE:
            prices_list = list(tick_buffer)
            variance_val = np.var(prices_list)
            x = np.arange(TICK_BUFFER_SIZE)
            slope_val, _, _, _, _ = linregress(x, prices_list)

            # Decide trend
            if slope_val > SLOPE_THRESHOLD:
                trend_str = "Bullish"
                order_signal = "BUY"
            elif slope_val < -SLOPE_THRESHOLD:
                trend_str = "Bearish"
                order_signal = "SELL"
            else:
                trend_str = "Neutral"
                order_signal = None

            # Log aggregated data in tick_data.csv
            date_agg = datetime.datetime.now().strftime("%Y-%m-%d")
            time_agg = datetime.datetime.now().strftime("%H:%M:%S")
            prices_str = ";".join(f"{p:.5f}" for p in prices_list)
            with open(TICK_CSV, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    date_agg,
                    time_agg,
                    SYMBOL,
                    prices_str,
                    f"{variance_val:.6f}",
                    f"{slope_val:.6f}",
                    trend_str
                ])

            logging.info("Order attempt #%d at %s", order_num, time_agg)
            logging.info("Aggregated 30 prices: %s", prices_str)
            logging.info("Computed Variance=%.6f, Slope=%.6f => Trend=%s", variance_val, slope_val, trend_str)

            # If slope indicates buy/sell, place an order
            if order_signal:
                current_price = prices_list[-1]
                result = place_market_order(SYMBOL, LOT_SIZE, order_signal, current_price, variance_val)
                time_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                if result is None:
                    logging.warning("%s order failed. retcode=N/A, comment=No result object. Check broker settings.",
                                    order_signal)
                elif result.retcode == mt5.TRADE_RETCODE_DONE:
                    profit_attr = getattr(result, "profit", 0.0)
                    logging.info("%s order placed successfully. retcode=%d, order=%d, deal=%d",
                                 order_signal, result.retcode, result.order, result.deal)
                    with open(ORDER_CSV, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            result.order,
                            order_num,
                            time_now,
                            SYMBOL,
                            order_signal,
                            LOT_SIZE,
                            f"{current_price:.5f}",
                            f"{profit_attr:.2f}",
                            f"Signal={order_signal}, Slope={slope_val:.4f}, Var={variance_val:.4f}"
                        ])
                else:
                    logging.warning("%s order failed. retcode=%s, comment=%s",
                                    order_signal, result.retcode, result.comment)
        else:
            logging.info("Tick buffer not full yet (%d/%d). Skipping order attempt.",
                         len(tick_buffer), TICK_BUFFER_SIZE)

        logging.info("Sleeping %d seconds before next iteration...\n", SLEEP_BETWEEN_ORDERS)
        time.sleep(SLEEP_BETWEEN_ORDERS)

    logging.info("Completed %d order attempts. Exiting script.", LOOP_ITERATIONS)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Script interrupted by user.")
    finally:
        mt5.shutdown()
