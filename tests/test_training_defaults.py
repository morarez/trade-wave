from ai.train_model import DEFAULT_TRAIN_SYMBOLS


def test_default_train_symbols_include_major_companies():
    assert len(DEFAULT_TRAIN_SYMBOLS) >= 10
    assert "AAPL" in DEFAULT_TRAIN_SYMBOLS
    assert "MSFT" in DEFAULT_TRAIN_SYMBOLS
    assert "GOOG" in DEFAULT_TRAIN_SYMBOLS
    assert "AMZN" in DEFAULT_TRAIN_SYMBOLS
    assert "NVDA" in DEFAULT_TRAIN_SYMBOLS
