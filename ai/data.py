
import pandas as pd
import numpy as np
from typing import Tuple, Union

from indicators import (
    add_sma,
    add_macd,
    add_rsi,
    add_stochastic,
    add_bollinger_bands,
    add_rolling_volatility,
    add_atr,
    add_roc,
    add_adx,
    add_obv,
    add_volume_features,
)


def features_for_series(series: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
    """Return a features DataFrame for a single price series or OHLCV DataFrame.

    If `series` is a Series, it's treated as `close` series (backwards-compatible).
    If `series` is a DataFrame, it should contain at least 'close' and optionally
    'high','low','volume' to compute richer features.
    """
    # Normalize input to DataFrame with named columns when possible
    if isinstance(series, pd.Series):
        df = pd.DataFrame({"close": series}).copy()
    else:
        df = series.copy()

    # Basic price-derived features
    if "close" in df.columns:
        df["ret_1"] = df["close"].pct_change()
        df["ret_5"] = df["close"].pct_change(5)
        df["ret_10"] = df["close"].pct_change(10)

    # Indicator enrichments (best-effort; keep original data if indicator fails)
    try:
        df = add_sma(df)
    except Exception:
        pass
    try:
        df = add_rsi(df)
    except Exception:
        pass
    try:
        df = add_macd(df)
    except Exception:
        pass
    try:
        df = add_stochastic(df)
    except Exception:
        pass
    try:
        df = add_bollinger_bands(df)
    except Exception:
        pass
    try:
        df = add_rolling_volatility(df)
    except Exception:
        pass
    try:
        df = add_atr(df)
    except Exception:
        pass
    try:
        df = add_roc(df)
    except Exception:
        pass
    try:
        df = add_adx(df)
    except Exception:
        pass
    try:
        df = add_obv(df)
    except Exception:
        pass
    try:
        df = add_volume_features(df)
    except Exception:
        pass

    # Final cleanup: replace infinities and drop rows with NaNs
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


def build_dataset(price_df: pd.DataFrame, target_horizon: int = 1) -> Tuple[pd.DataFrame, pd.Series]:
    """Build a stacked dataset across symbols.

    X: features at time t
    y: next-period return (t -> t+target_horizon)
    """
    X_parts = []
    y_parts = []
    for symbol in price_df.columns:
        series = price_df[symbol].astype(float).copy()
        feats = features_for_series(series)
        # compute future return as target
        target = series.pct_change(periods=target_horizon).shift(-target_horizon)
        target = target.reindex(feats.index)
        df = feats.copy()
        df["symbol"] = symbol
        X_parts.append(df)
        y_parts.append(target)

    X = pd.concat(X_parts, axis=0)
    y = pd.concat(y_parts, axis=0)

    # drop any leftover NaNs
    valid = ~y.isna()
    X = X.loc[valid]
    y = y.loc[valid]

    return X, y


def make_features_for_backtest(series: pd.Series) -> pd.DataFrame:
    """Create features aligned to the backtest input `df` (single symbol).

    Returns features indexed the same as input series (may have NaNs at start).
    """
    return features_for_series(series)