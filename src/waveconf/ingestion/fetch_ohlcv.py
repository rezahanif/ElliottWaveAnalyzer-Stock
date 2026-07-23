import os
import json
from datetime import datetime
import pandas as pd
import yfinance as yf
import ccxt
import urllib3

# Suppress unverified HTTPS warnings from urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def fetch_historical_btc_1d():
    print("--- Phase 1 (1D): Fetching Historic Daily Data (2011-2020) via Yahoo Finance ---")
    ticker = yf.Ticker("BTC-USD")
    yf_df = ticker.history(start="2011-01-01", end="2020-01-01", interval="1d")
    
    yf_data = []
    for index, row in yf_df.iterrows():
        ts_ms = int(index.timestamp() * 1000)
        yf_data.append([
            ts_ms,
            round(float(row['Open']), 2),
            round(float(row['High']), 2),
            round(float(row['Low']), 2),
            round(float(row['Close']), 2),
            round(float(row['Volume']), 2)
        ])
    print(f"Loaded {len(yf_data)} daily candles from Yahoo Finance.")

    print("\n--- Phase 2 (1D): Fetching Modern Precision Daily Data (2020-Now) via Binance ---")
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'verify': False,  # Bypasses local SSL certificate issues on some systems
    })
    
    since = exchange.parse8601('2020-01-01T00:00:00Z')
    binance_data = []
    
    while since < exchange.milliseconds():
        try:
            candles = exchange.fetch_ohlcv('BTC/USDT', timeframe='1d', since=since, limit=1000)
            if not candles:
                break
            
            print(f"Fetched batch from: {datetime.utcfromtimestamp(candles[0][0]/1000).strftime('%Y-%m-%d')}")
            binance_data.extend(candles)
            since = candles[-1][0] + 86400000 
        except Exception as e:
            print(f"Error fetching from Binance: {e}")
            break

    # If Binance fetch failed or returned nothing (e.g. due to ISP blocks), fall back to Yahoo Finance
    if not binance_data:
        print("Falling back to Yahoo Finance for modern (2020-Now) daily data...")
        try:
            yf_df_modern = ticker.history(start="2020-01-01", end=datetime.now().strftime("%Y-%m-%d"), interval="1d")
            for index, row in yf_df_modern.iterrows():
                ts_ms = int(index.timestamp() * 1000)
                binance_data.append([
                    ts_ms,
                    round(float(row['Open']), 2),
                    round(float(row['High']), 2),
                    round(float(row['Low']), 2),
                    round(float(row['Close']), 2),
                    round(float(row['Volume']), 2)
                ])
            print(f"Loaded {len(binance_data)} daily candles from Yahoo Finance (fallback).")
        except Exception as fallback_e:
            print(f"Error fetching fallback daily data: {fallback_e}")

    print(f"Loaded {len(binance_data)} daily candles for the modern period.")

    print("\n--- Phase 3 (1D): Stitching Data Layers Together ---")
    columns = ["timestamp_ms", "open", "high", "low", "close", "volume"]
    df_early = pd.DataFrame(yf_data, columns=columns)
    df_modern = pd.DataFrame(binance_data, columns=columns)
    
    # Avoid deprecation warnings by filtering out empty dataframes
    dfs_to_concat = [df for df in [df_early, df_modern] if not df.empty]
    if dfs_to_concat:
        df_combined = pd.concat(dfs_to_concat).drop_duplicates(subset=['timestamp_ms'], keep='last')
    else:
        df_combined = pd.DataFrame(columns=columns)
        
    df_combined = df_combined.sort_values(by='timestamp_ms').reset_index(drop=True)

    final_output = {
        "asset": "BTCUSD",
        "timeframe": "1D",
        "columns": columns,
        "data": df_combined.values.tolist()
    }

    os.makedirs('data/ohlcv', exist_ok=True)
    output_path = 'data/ohlcv/BTC_1D.json'
    with open(output_path, 'w') as f:
        json.dump(final_output, f)
        
    print(f"Success! Saved {len(df_combined)} total 1D candles to {output_path}\n")


def fetch_historical_btc_4h():
    print("--- Fetching BTC/USDT 4H candles from Binance (since 2017-08-17) ---")
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'verify': False, 
    })
    
    # Optional: Configure proxy if your ISP blocks Binance (e.g. Internet Positif)
    # exchange.proxies = {
    #     'http': 'http://127.0.0.1:7890',
    #     'https': 'http://127.0.0.1:7890',
    # }

    since = exchange.parse8601('2017-08-17T00:00:00Z') # Binance launch date
    binance_data = []
    
    while since < exchange.milliseconds():
        try:
            candles = exchange.fetch_ohlcv('BTC/USDT', timeframe='4h', since=since, limit=1000)
            if not candles:
                break
            
            first_date = datetime.utcfromtimestamp(candles[0][0]/1000).strftime('%Y-%m-%d %H:%M')
            last_date = datetime.utcfromtimestamp(candles[-1][0]/1000).strftime('%Y-%m-%d %H:%M')
            print(f"Fetched {len(candles)} candles from {first_date} to {last_date}")
            
            binance_data.extend(candles)
            since = candles[-1][0] + (4 * 60 * 60 * 1000)
        except Exception as e:
            print(f"Error fetching from Binance: {e}")
            print("\n[TIP] If you get 403 / certificate errors, your ISP is blocking Binance.")
            print("Please enable a VPN or configure 'exchange.proxies' in this script and try again.")
            break

    if not binance_data:
        print("No 4H data fetched. Make sure your VPN is on or check your network.")
        return

    columns = ["timestamp_ms", "open", "high", "low", "close", "volume"]
    df = pd.DataFrame(binance_data, columns=columns)
    df = df.drop_duplicates(subset=['timestamp_ms']).sort_values(by='timestamp_ms').reset_index(drop=True)
    
    final_output = {
        "asset": "BTCUSD",
        "timeframe": "4H",
        "columns": columns,
        "data": df.values.tolist()
    }
    
    os.makedirs('data/ohlcv', exist_ok=True)
    output_path = 'data/ohlcv/BTC_4H.json'
    with open(output_path, 'w') as f:
        json.dump(final_output, f)
        
    print(f"Success! Saved {len(df)} total 4H candles to {output_path}\n")

def fetch_historical_btc_1w():
    print("--- Fetching BTC/USDT 1W candles from Binance (since 2017-08-17) ---")
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'verify': False, 
    })
    
    since = exchange.parse8601('2017-08-17T00:00:00Z') # Binance launch date
    binance_data = []
    
    while since < exchange.milliseconds():
        try:
            candles = exchange.fetch_ohlcv('BTC/USDT', timeframe='1w', since=since, limit=1000)
            if not candles:
                break
            
            first_date = datetime.utcfromtimestamp(candles[0][0]/1000).strftime('%Y-%m-%d %H:%M')
            last_date = datetime.utcfromtimestamp(candles[-1][0]/1000).strftime('%Y-%m-%d %H:%M')
            print(f"Fetched {len(candles)} candles from {first_date} to {last_date}")
            
            binance_data.extend(candles)
            since = candles[-1][0] + (7 * 24 * 60 * 60 * 1000)
        except Exception as e:
            print(f"Error fetching from Binance: {e}")
            print("\n[TIP] If you get 403 / certificate errors, your ISP is blocking Binance.")
            break

    if not binance_data:
        print("No 1W data fetched.")
        return

    columns = ["timestamp_ms", "open", "high", "low", "close", "volume"]
    df = pd.DataFrame(binance_data, columns=columns)
    df = df.drop_duplicates(subset=['timestamp_ms']).sort_values(by='timestamp_ms').reset_index(drop=True)
    
    final_output = {
        "asset": "BTCUSD",
        "timeframe": "1W",
        "columns": columns,
        "data": df.values.tolist()
    }
    
    os.makedirs('data/ohlcv', exist_ok=True)
    output_path = 'data/ohlcv/BTC_1W.json'
    with open(output_path, 'w') as f:
        json.dump(final_output, f)
        
    print(f"Success! Saved {len(df)} total 1W candles to {output_path}\n")

if __name__ == "__main__":
    # Fetch 1W data (Yahoo Finance)
    fetch_historical_btc_1w()
