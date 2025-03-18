#!/usr/bin/env python3

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
SYMBOL = "BTCUSD"               # Adjust as needed (e.g., "BTCUSD.pro" if required by broker)
TICK_BUFFER_SIZE = 30           # Rolling buffer size for statistics
TICKS_PER_CALL = 200            # Maximum ticks to fetch per call
LOOKBACK_MINUTES = 2            # Start time: 2 minutes in the past
LOOP_ITERATIONS = 100           # Number of order attempts
SLEEP_BETWEEN_ORDERS = 30       # Seconds between each order attempt

SLOPE_THRESHOLD = 0.0001        # Slope threshold for trade signal
LOT_SIZE = 0.01                 # Trading volume

# Variance Floor: if calculated variance < 30, then use 30
VARIANCE_FLOOR = 30.0

# CSV file names
RAW_TICK_CSV = "raw_tick_data.csv"  # Continuously store every tick
TICK_CSV = "tick_data.csv"          # Store aggregated data (last 30 ticks, variance, slope, trend)
ORDER_CSV = "order_data.csv"        # Store order request data

# -------------------------------
# CSV Utilities for Raw Tick Data
# -------------------------------
def prepare_raw_tick_csv():
    if not os.path.exists(RAW_TICK_CSV):
        with open(RAW_TICK_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["date", "time", "symbol", "price"])

def append_raw_tick_csv_row(date_str, time_str, symbol, price):
    with open(RAW_TICK_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([date_str, time_str, symbol, f"{price:.5f}"])

# -------------------------------
# CSV Utilities for Aggregated Tick Data
# -------------------------------
def prepare_tick_csv():
    if not os.path.exists(TICK_CSV):
        with open(TICK_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["date", "time", "symbol", "prices", "variance", "slope", "trend"])

def append_tick_csv_row(date_str, time_str, symbol, prices_str, variance, slope, trend):
    with open(TICK_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([date_str, time_str, symbol, prices_str, variance, slope, trend])

# -------------------------------
# CSV Utilities for Order Data
# -------------------------------
def prepare_order_csv():
    if not os.path.exists(ORDER_CSV):
        with open(ORDER_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Ticket", "Order", "Time", "Symbol", "Type", "Volume", "Price", "Profit", "Comment"])

def append_order_csv_row(ticket, order_num, time_str, symbol, order_type, volume, price, profit, comment):
    with open(ORDER_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([ticket, order_num, time_str, symbol, order_type, volume, f"{price:.5f}", f"{profit:.2f}", comment])

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

if not mt5.symbol_select(SYMBOL, True):
    logging.error("symbol_select(%s) failed. Check symbol name or broker availability.", SYMBOL)
    mt5.shutdown()
    quit()

# Check for auto trading enabled (warn if not)
terminal_info = mt5.terminal_info()
if terminal_info is None or not terminal_info.trade_allowed:
    logging.warning("AutoTrading is disabled in your MT5 terminal. Please enable it for order execution.")

# -------------------------------
# Global Variables: Tick Buffer and Time Tracking
# -------------------------------
tick_buffer = deque(maxlen=TICK_BUFFER_SIZE)
start_time = datetime.datetime.now() - datetime.timedelta(minutes=LOOKBACK_MINUTES)
latest_tick_time = start_time
previous_slope = 0.0

# -------------------------------
# Function: Fetch New Ticks
# -------------------------------
def fetch_new_ticks(symbol, from_time):
    """
    Fetch up to TICKS_PER_CALL new ticks from 'from_time' onward.
    """
    ticks = mt5.copy_ticks_from(symbol, from_time, TICKS_PER_CALL, mt5.COPY_TICKS_ALL)
    return ticks

# -------------------------------
# Function: Place Market Order
# -------------------------------
def place_market_order(symbol, volume, order_type, price, variance):
    """
    Places a market order using variance-based stops.
    Enforces a variance floor: if computed variance < VARIANCE_FLOOR, use VARIANCE_FLOOR.
    
    For BUY: TP = price + effective_variance, SL = price - effective_variance.
    For SELL: TP = price - effective_variance, SL = price + effective_variance.
    """
    effective_variance = variance if variance >= VARIANCE_FLOOR else VARIANCE_FLOOR

    if order_type == "BUY":
        tp = price + effective_variance
        sl = price - effective_variance
        mt5_order_type = mt5.ORDER_TYPE_BUY
        comment = f"Slope-based Buy (var={effective_variance:.2f})"
    else:  # SELL
        tp = price - effective_variance
        sl = price + effective_variance
        mt5_order_type = mt5.ORDER_TYPE_SELL
        comment = f"Slope-based Sell (var={effective_variance:.2f})"

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
# Main Trading Loop
# -------------------------------
def main():
    global latest_tick_time, previous_slope

    logging.info("Starting mechanical HFT agent for %s", SYMBOL)
    logging.info("Will attempt up to %d trades, every %d seconds.", LOOP_ITERATIONS, SLEEP_BETWEEN_ORDERS)
    logging.info("Initial start time: %s", latest_tick_time.strftime("%Y-%m-%d %H:%M:%S"))

    for order_num in range(1, LOOP_ITERATIONS + 1):
        # 1) Fetch new ticks (do not print every tick to terminal; raw ticks are stored in CSV)
        ticks = fetch_new_ticks(SYMBOL, latest_tick_time)
        if ticks is not None and len(ticks) > 0:
            for tick in ticks:
                tick_dt = datetime.datetime.fromtimestamp(tick["time"])
                if tick_dt > latest_tick_time:
                    latest_tick_time = tick_dt + datetime.timedelta(microseconds=1)

                price = tick["last"]
                if price <= 0:
                    price = (tick["bid"] + tick["ask"]) / 2

                # Append each tick to raw_tick_data.csv
                date_str = tick_dt.strftime("%Y-%m-%d")
                time_str = tick_dt.strftime("%H:%M:%S.%f")[:-3]
                append_raw_tick_csv_row(date_str, time_str, SYMBOL, price)

                # Append tick to the rolling buffer
                tick_buffer.append(price)
        else:
            logging.info("No new ticks or error: %s", mt5.last_error())

        # 2) Every 30 seconds, if tick_buffer is full, compute indicators & attempt order
        if len(tick_buffer) == TICK_BUFFER_SIZE:
            prices_list = list(tick_buffer)
            variance_val = np.var(prices_list)
            slope_val, _, _, _, _ = linregress(range(TICK_BUFFER_SIZE), prices_list)

            # Determine trend for aggregated data
            if slope_val > SLOPE_THRESHOLD:
                trend_decision = "Bullish"
            elif slope_val < -SLOPE_THRESHOLD:
                trend_decision = "Bearish"
            else:
                trend_decision = "Neutral"

            # Prepare aggregated tick data for tick_data.csv:
            prices_str = ";".join(f"{p:.5f}" for p in prices_list)
            # Use current time from last tick in buffer for aggregation timestamp
            agg_time = datetime.datetime.now()
            date_str = agg_time.strftime("%Y-%m-%d")
            time_str = agg_time.strftime("%H:%M:%S")
            append_tick_csv_row(date_str, time_str, SYMBOL, prices_str,
                                f"{variance_val:.6f}", f"{slope_val:.6f}", trend_decision)

            logging.info("Order attempt #%d at %s", order_num, agg_time.strftime("%H:%M:%S"))
            logging.info("Aggregated 30 prices: %s", prices_str)
            logging.info("Computed Variance=%.6f, Slope=%.6f", variance_val, slope_val)

            # 3) Determine order signal: BUY if bullish, SELL if bearish
            order_signal = None
            if trend_decision == "Bullish":
                order_signal = "BUY"
            elif trend_decision == "Bearish":
                order_signal = "SELL"
            else:
                logging.info("No trade signal triggered (trend=Neutral).")

            previous_slope = slope_val

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
                    append_order_csv_row(
                        ticket=result.order,
                        order_num=order_num,
                        time_str=time_now,
                        symbol=SYMBOL,
                        order_type=order_signal,
                        volume=LOT_SIZE,
                        price=current_price,
                        profit=profit_attr,
                        comment=f"Signal={order_signal}, Slope={slope_val:.4f}, VarUsed={max(variance_val, VARIANCE_FLOOR):.4f}"
                    )
                else:
                    logging.warning("%s order failed. retcode=%s, comment=%s",
                                    order_signal,
                                    result.retcode,
                                    result.comment)
        else:
            logging.info("Tick buffer not full yet (%d/%d). Skipping order attempt.",
                         len(tick_buffer), TICK_BUFFER_SIZE)

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