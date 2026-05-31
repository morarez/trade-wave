
import pandas as pd
import numpy as np
from typing import Tuple

from indicators.moving_averages import add_sma, add_macd
from indicators.momentum import add_rsi


def features_for_series(series: pd.Series) -> pd.DataFrame:
    """Return a features DataFrame for a single price series.

    Features are aligned to time t and the target is the next-period return.
    """
    df = pd.DataFrame({"close": series}).copy()
    # Basic price-derived features
    df["ret_1"] = df["close"].pct_change()
    df["ret_5"] = df["close"].pct_change(5)
    df["ret_10"] = df["close"].pct_change(10)

    # Technical indicators (these functions add columns in-place)
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

    # Fill/clean
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