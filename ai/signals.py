import pandas as pd
from prophet import Prophet

def generate_forecast(df, periods=10, freq="H"):
    """
    Generate a Prophet forecast from OHLCV data.

    df: pandas DataFrame with columns ["open_time", "close"]
    periods: int - number of future steps to forecast
    freq: str - frequency of prediction ("H"=hourly, "D"=daily, etc.)
    """
    if "open_time" not in df.columns or "close" not in df.columns:
        raise ValueError("DataFrame must have 'open_time' and 'close' columns")

    # Ensure numeric
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"])

    # Prepare Prophet dataframe
    prophet_df = df[["open_time", "close"]].rename(
        columns={"open_time": "ds", "close": "y"}
    )

    # Remove timezone if present
    if pd.api.types.is_datetime64tz_dtype(prophet_df["ds"]):
        prophet_df["ds"] = prophet_df["ds"].dt.tz_convert(None)

    model = Prophet(daily_seasonality=True)
    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)

    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
