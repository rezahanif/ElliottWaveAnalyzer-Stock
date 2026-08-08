from src.btc.wave_model.schema import get_btc_schema
from src.stock.features.schema import get_default_schema


def test_btc_and_stock_scalers_are_independent():
    btc = get_btc_schema()
    stock = get_default_schema()
    btc_scalers = btc.scalers_for(["open_norm", "close_norm"])
    stock_scalers = stock.scalers_for(["rsi_14", "macd"])
    assert btc is not stock
    assert set(btc_scalers) != set(stock_scalers)
    for name in btc_scalers:
        assert all(btc_scalers[name] is not value for value in stock_scalers.values())


def test_scaler_calls_return_fresh_objects():
    schema = get_default_schema()
    first = schema.scalers_for(["rsi_14"])
    second = schema.scalers_for(["rsi_14"])
    assert first["rsi_14"] is not second["rsi_14"]
    assert first["rsi_14"].__class__ is second["rsi_14"].__class__
    assert first["rsi_14"].__class__.__module__ == "sklearn.preprocessing._data"


if __name__ == "__main__":
    test_btc_and_stock_scalers_are_independent()
    test_scaler_calls_return_fresh_objects()
    print("ok")
