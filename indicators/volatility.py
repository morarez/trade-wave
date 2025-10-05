import pandas_ta as ta

def add_bollinger_bands(df, length=20, std=2):
    bb = ta.bbands(df["close"], length=length, std=std)
    df = df.join(bb)
    return df
