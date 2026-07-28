"""
Order-book conviction layer for the BTC Elliott Wave pipeline.

Pulls L2 order-book snapshots from Binance, Coinbase, and OKX via CCXT,
detects USD-notional walls (>= wall_threshold_usd from config/orderbook.yaml),
and derives a conviction multiplier applied to the calendar-adjusted
confluence strength.

Public API:
    from src.btc.orderbook.fetcher import fetch_multi_exchange_orderbook
    from src.btc.orderbook.scorer import OrderBookConvictionScorer, ConvictionReport
    from src.btc.orderbook.snapshot import write_snapshot, load_latest_snapshot
"""
