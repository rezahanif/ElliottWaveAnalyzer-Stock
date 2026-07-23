import os
import json
import sys
import pandas as pd

def generate_volatility_layers(input_path='data/ohlcv/BTC_4H.json'):
    if not os.path.exists(input_path):
        print(f"Error: Could not find raw data at {input_path}. Please run your fetcher script first.")
        return

    # 1. Load the raw JSON data
    with open(input_path, 'r') as f:
        raw_json = json.load(f)
    
    # 2. Parse into a clean Pandas DataFrame
    df = pd.DataFrame(raw_json['data'], columns=raw_json['columns'])
    
    # 3. Detect timeframe to dynamically adjust lookback periods to represent actual calendar days
    timeframe = raw_json.get('timeframe', '1D').upper()
    if timeframe == '4H':
        # 1 Day = 6 candles of 4H
        atr_20_periods = 20 * 6  # 120 periods representing 20 calendar days
        atr_14_periods = 14 * 6  # 84 periods representing 14 calendar days
        date_format = '%Y-%m-%d %H:%M'
    else:
        atr_20_periods = 20
        atr_14_periods = 14
        date_format = '%Y-%m-%d'

    # Convert timestamps back to human dates with correct timeframe resolution
    df['date'] = pd.to_datetime(df['timestamp_ms'], unit='ms').dt.strftime(date_format)

    # 4. Calculate True Range (TR) - The foundation for both layers
    df['prev_close'] = df['close'].shift(1)
    df['h_l'] = df['high'] - df['low']
    df['h_pc'] = (df['high'] - df['prev_close']).abs()
    df['l_pc'] = (df['low'] - df['prev_close']).abs()
    df['true_range'] = df[['h_l', 'h_pc', 'l_pc']].max(axis=1)

    # 5. Track A: The "Wall Street / Institutional" Layer (20-Day calendar lookback)
    df['atr_20'] = df['true_range'].ewm(alpha=1/atr_20_periods, adjust=False).mean()
    df['wall_street_threshold_pct'] = round((df['atr_20'] * 3 / df['close']) * 100, 2)

    # 6. Track B: The "Behavioral / Cyclic" Layer (14-Day calendar lookback)
    df['atr_14'] = df['true_range'].ewm(alpha=1/atr_14_periods, adjust=False).mean()
    df['behavioral_threshold_pct'] = round((df['atr_14'] * 1.5 / df['close']) * 100, 2)

    # Clean up calculation columns before exporting
    df_clean = df[['timestamp_ms', 'date', 'open', 'high', 'low', 'close', 'volume', 
                   'wall_street_threshold_pct', 'behavioral_threshold_pct']].dropna()

    # 7. Structure and Save Final Outputs
    output_data = {
        "asset": raw_json['asset'],
        "timeframe": timeframe,
        "metrics_description": {
            "wall_street_threshold_pct": f"Dynamic threshold based on 20-Day rolling calendar liquidity cycles ({atr_20_periods} periods on {timeframe}).",
            "behavioral_threshold_pct": f"Dynamic threshold based on 14-Day half-lunar emotional/algorithmic patterns ({atr_14_periods} periods on {timeframe})."
        },
        "columns": list(df_clean.columns),
        "data": df_clean.values.tolist()
    }

    os.makedirs('data/pivots', exist_ok=True)
    output_path = f'data/pivots/BTC_{timeframe}_with_layers.json'
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Success! Processed {len(df_clean)} rows of {timeframe} data.")
    print(f"Saved structural dual-layer matrix to: {output_path}")
    
    # Quick live preview of the latest calculations
    print("\n--- Recent Matrix Preview ---")
    print(df_clean[['date', 'close', 'wall_street_threshold_pct', 'behavioral_threshold_pct']].tail(5).to_string(index=False))

if __name__ == "__main__":
    # Default to 4H but allow passing custom paths via CLI
    path = sys.argv[1] if len(sys.argv) > 1 else 'data/ohlcv/BTC_4H.json'
    generate_volatility_layers(path)