#!/usr/bin/env python3
"""
Mechanical HFT Agent with Weissman-Inspired Stops & Extended Tick Logging

Features:
1) Repeatedly fetches new ticks from MetaTrader 5 for a chosen symbol.
2) Maintains a rolling buffer (deque) of the last 30 prices.
3) Computes variance & slope (via linear regression) once the buffer is full.
4) Uses a "Weissman-like" volatility-based stop:
   - defaultStop = 2 * stdev(prices)
   - ensures SL/TP are on the correct side of the price for BUY/SELL.
   - compares with brokerMinStop if needed.
5) Every 30 seconds, if slope > +threshold => place BUY; if slope < -threshold => place SELL.
   - SL & TP are set using the "Weissman-like" approach to avoid "Invalid stops."
6) Logs every tick to tick_data.csv with columns:
   [date, time, symbol, price, variance, slope, trend].
   - If slope/variance not computed yet, store "N/A".
7) Logs each order to order_data.csv with columns:
   [Ticket, Order, Time, Symbol, Type, Volume, Price, Profit, Comment].

Author: [Your Name]
Pretend: A professional programmer with a Harvard degree and 10 years of experience.
"""

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
SYMBOL = "BTCUSD"                # Adjust if your broker uses a different symbol, e.g. "BTCUSD.pro"
TICK_BUFFER_SIZE = 30            # Number of ticks in the rolling buffer
TICKS_PER_CALL = 200             # Max ticks per fetch
LOOKBACK_MINUTES = 2             # Start 2 minutes in the past
LOOP_ITERATIONS = 100            # Max order attempts
SLEEP_BETWEEN_ORDERS = 30        # Seconds between order attempts

# Slope threshold for bullish/bearish signals
SLOPE_THRESHOLD = 0.0001

# Trading parameters
LOT_SIZE = 0.01

# "Weissman-like" approach: 2 × stdev for stop
STDEV_MULTIPLIER = 2.0

# If your broker has a min stop distance, define it or fetch from symbol_info.
# For demonstration, we set it to 0.0005 or fetch dynamically below.
MIN_STOP_DEFAULT = 0.0005

# CSV Files
TICK_CSV_FILE = "tick_data.csv"
ORDER_CSV_FILE = "order_data.csv"

# Deque for the last 30 prices
tick_buffer = deque(maxlen=TICK_BUFFER_SIZE)

# We'll track the last slope to see if there's a crossing from negative to positive or vice versa
previous_slope = 0.0

# For fetching new ticks
start_time = datetime.datetime.now() - datetime.timedelta(minutes=LOOKBACK_MINUTES)
latest_tick_time = start_time

# -------------------------------
# CSV: Tick Data
# -------------------------------
def prepare_tick_csv():
    """
    Prepare tick_data.csv with columns:
    date, time, symbol, price, variance, slope, trend
    """
    with open(TICK_CSV_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["date", "time", "symbol", "price", "variance", "slope", "trend"])

def append_tick_csv_row(date_str, time_str, symbol, price, variance, slope, trend):
    """
    Append a row to tick_data.csv. If slope/variance not computed yet, pass "N/A".
    """
    with open(TICK_CSV_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            date_str,
            time_str,
            symbol,
            f"{price:.5f}",
            f"{variance}" if isinstance(variance, float) else variance,
            f"{slope}" if isinstance(slope, float) else slope,
            trend
        ])

# -------------------------------
# CSV: Order Data
# -------------------------------
def prepare_order_csv():
    """
    Prepare order_data.csv with columns:
    Ticket, Order, Time, Symbol, Type, Volume, Price, Profit, Comment
    """
    with open(ORDER_CSV_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Ticket", "Order", "Time", "Symbol", "Type", "Volume", "Price", "Profit", "Comment"])

def append_order_csv_row(ticket, order_num, time_str, symbol, order_type, volume, price, profit, comment):
    """
    Append a row to order_data.csv.
    """
    with open(ORDER_CSV_FILE, 'a', newline='') as f:
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
prepare_tick_csv()
prepare_order_csv()

# -------------------------------
# Initialize MT5
# -------------------------------
if not mt5.initialize():
    logging.error("MT5 initialize() failed, error code = %s", mt5.last_error())
    quit()

# Ensure the symbol is recognized
if not mt5.symbol_select(SYMBOL, True):
    logging.error("symbol_select(%s) failed. Check symbol name or broker availability.", SYMBOL)
    mt5.shutdown()
    quit()

# (Optional) fetch brokerMinStop from symbol_info
symbol_info = mt5.symbol_info(SYMBOL)
if symbol_info is None:
    logging.error("Failed to get symbol info for %s", SYMBOL)
    mt5.shutdown()
    quit()
broker_min_stop_points = symbol_info.trade_stops_level  # e.g. 50 points => 5 pips if point=0.1 pip
broker_min_stop_price = broker_min_stop_points * symbol_info.point
if broker_min_stop_price <= 0:
    broker_min_stop_price = MIN_STOP_DEFAULT  # fallback if brokerMinStop is 0

logging.info("Broker min stop distance (price) ~ %.5f", broker_min_stop_price)

# -------------------------------
# Tick Fetch
# -------------------------------
def fetch_new_ticks(symbol, from_time):
    """
    Fetch up to TICKS_PER_CALL new ticks from 'from_time' onward.
    """
    ticks = mt5.copy_ticks_from(symbol, from_time, TICKS_PER_CALL, mt5.COPY_TICKS_ALL)
    return ticks

# -------------------------------
# Place Market Order
# -------------------------------
def place_market_order(symbol, volume, order_type, price, stop_dist):
    """
    Places a market order (BUY or SELL) with a given stop_dist.
    For a BUY, we do: TP = price + stop_dist, SL = price - stop_dist
    For a SELL, we do: TP = price - stop_dist, SL = price + stop_dist
    Returns the result from order_send.
    """
    if order_type == "BUY":
        tp = price + stop_dist
        sl = price - stop_dist
        mt5_order_type = mt5.ORDER_TYPE_BUY
        comment = f"Slope-based Buy (stop={stop_dist:.5f})"
    else:
        tp = price - stop_dist
        sl = price + stop_dist
        mt5_order_type = mt5.ORDER_TYPE_SELL
        comment = f"Slope-based Sell (stop={stop_dist:.5f})"

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
        "type_filling": mt5.ORDER_FILLING_IOC,
        "comment": comment
    }

    result = mt5.order_send(request)
    return result

# -------------------------------
# Main Loop
# -------------------------------
def main():
    global latest_tick_time, previous_slope

    logging.info("Starting mechanical HFT agent for %s", SYMBOL)
    logging.info("Attempting up to %d trades, every %d seconds", LOOP_ITERATIONS, SLEEP_BETWEEN_ORDERS)
    logging.info("Initial start time: %s", latest_tick_time.strftime("%Y-%m-%d %H:%M:%S"))

    for order_num in range(1, LOOP_ITERATIONS + 1):
        # 1) Fetch new ticks
        ticks = fetch_new_ticks(SYMBOL, latest_tick_time)
        if ticks is not None and len(ticks) > 0:
            logging.info("Fetched %d new ticks since %s", len(ticks), latest_tick_time.strftime("%H:%M:%S"))
            for tick in ticks:
                tick_dt = datetime.datetime.fromtimestamp(tick["time"])
                if tick_dt > latest_tick_time:
                    latest_tick_time = tick_dt + datetime.timedelta(microseconds=1)

                # Decide price
                price = tick["last"]
                if price <= 0:
                    price = (tick["bid"] + tick["ask"]) / 2

                # Append to buffer
                tick_buffer.append(price)

                # For CSV logging, we compute or reuse the last slope/variance/trend if available
                if len(tick_buffer) < TICK_BUFFER_SIZE:
                    variance_str = "N/A"
                    slope_str = "N/A"
                    trend_str = "N/A"
                else:
                    # We have enough ticks to compute
                    prices_list = list(tick_buffer)
                    # Variance is standard dev squared
                    variance_val = np.var(prices_list)
                    # slope
                    slope_val, _, _, _, _ = linregress(range(TICK_BUFFER_SIZE), prices_list)
                    # Decide trend
                    if slope_val > SLOPE_THRESHOLD:
                        trend_str = "Bullish"
                    elif slope_val < -SLOPE_THRESHOLD:
                        trend_str = "Bearish"
                    else:
                        trend_str = "Neutral"

                    variance_str = f"{variance_val:.6f}"
                    slope_str = f"{slope_val:.6f}"

                # Append row to tick_data.csv
                date_str = tick_dt.strftime("%Y-%m-%d")
                time_str = tick_dt.strftime("%H:%M:%S.%f")[:-3]
                append_tick_csv_row(date_str, time_str, SYMBOL, price, variance_str, slope_str, trend_str)
                logging.info("Tick at %s: price=%.5f", time_str, price)
        else:
            logging.info("No new ticks or error: %s", mt5.last_error())

        # 2) Attempt order if we have a full buffer
        if len(tick_buffer) == TICK_BUFFER_SIZE:
            prices_list = list(tick_buffer)
            # Compute standard deviation
            std_val = np.std(prices_list)
            # "Weissman-like" stop distance => max(2 * std, broker_min_stop_price)
            default_stop_dist = STDEV_MULTIPLIER * std_val
            # Compare with broker's min stop distance
            effective_stop = max(default_stop_dist, broker_min_stop_price)

            # Slope
            slope_val, _, _, _, _ = linregress(range(TICK_BUFFER_SIZE), prices_list)

            # Print for debugging
            logging.info("Order attempt #%d at %s", order_num, datetime.datetime.now().strftime("%H:%M:%S"))
            logging.info("Last 30 prices: %s", " ".join(f"{p:.5f}" for p in prices_list))
            logging.info("Std=%.6f => default_stop=%.6f => effective_stop=%.6f, slope=%.6f",
                         std_val, default_stop_dist, effective_stop, slope_val)

            # Check slope crossing
            current_price = prices_list[-1]
            order_signal = None
            if slope_val > SLOPE_THRESHOLD:
                order_signal = "BUY"
            elif slope_val < -SLOPE_THRESHOLD:
                order_signal = "SELL"
            else:
                logging.info("No slope-based trade triggered (slope=%.6f).", slope_val)

            if order_signal:
                result = place_market_order(SYMBOL, LOT_SIZE, order_signal, current_price, effective_stop)
                time_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                    logging.info("%s order placed successfully. retcode=%d, order=%d, deal=%d",
                                 order_signal, result.retcode, result.order, result.deal)
                    # Log to order_data.csv
                    append_order_csv_row(
                        ticket=result.order,
                        order_num=order_num,
                        time_str=time_now,
                        symbol=SYMBOL,
                        order_type=order_signal,
                        volume=LOT_SIZE,
                        price=current_price,
                        profit=result.profit,
                        comment=f"Slope={slope_val:.4f}, stop={effective_stop:.4f}"
                    )
                else:
                    # If it fails, check result.comment or retcode
                    logging.warning("%s order failed. retcode=%s, comment=%s",
                                    order_signal,
                                    result.retcode if result else "N/A",
                                    result.comment if result else "No result object")
        else:
            logging.info("Tick buffer not full (%d/%d), skipping order attempt.", len(tick_buffer), TICK_BUFFER_SIZE)

        logging.info("Sleeping %d seconds before next iteration...\n", SLEEP_BETWEEN_ORDERS)
        time.sleep(SLEEP_BETWEEN_ORDERS)

    logging.info("Completed %d order attempts. Exiting script.", LOOP_ITERATIONS)

# -------------------------------
# Entry Point
# -------------------------------
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Script interrupted by user.")
    finally:
        mt5.shutdown()
