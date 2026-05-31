import pandas as pd
from indicators.moving_averages import add_macd
from strategies.base import Strategy


class MACDTrend(Strategy):
    name = "macd_trend"

    def apply_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = add_macd(df)
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        df = self.apply_indicators(df)
        signals = pd.Series("HOLD", index=df.index)

        macd = pd.to_numeric(df["MACD_12_26_9"], errors="coerce")
        macds = pd.to_numeric(df["MACDs_12_26_9"], errors="coerce")

        buy = macd > macds
        sell = macd < macds
        buy = buy.fillna(False)
        sell = sell.fillna(False)

        signals.loc[buy] = "BUY"
        signals.loc[sell] = "SELL"
        return signals

    def generate_signal(self, df: pd.DataFrame) -> str:
        df = self.apply_indicators(df)
        latest = df.iloc[-1]
        if latest["MACD_12_26_9"] > latest["MACDs_12_26_9"]:
            return "BUY"
        elif latest["MACD_12_26_9"] < latest["MACDs_12_26_9"]:
            return "SELL"
        else:
            return "HOLD"


strategy = MACDTrend()
