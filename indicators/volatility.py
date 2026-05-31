import pandas_ta as ta

def add_bollinger_bands(df, length=20, std=2):
    """Add Bollinger Bands columns to DataFrame.
    
    Computes and adds upper band, middle band, and lower band columns.
    
    Args:
        df: DataFrame with 'close' column.
        length: lookback period for Bollinger Bands (default 20).
        std: number of standard deviations for band width (default 2).
    
    Returns:
        DataFrame with Bollinger Bands columns added.
    """
    bb = ta.bbands(df["close"], length=length, std=std)
    df = df.join(bb)
    return df
