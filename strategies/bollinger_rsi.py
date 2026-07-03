import pandas as pd
from indicators.momentum import add_rsi
from indicators.volatility import add_bollinger_bands
from strategies.base import Strategy


class BollingerRSI(Strategy):
    name = "bollinger_rsi"

    def apply_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = add_bollinger_bands(df, length=20)
        df = add_rsi(df)
        return df

    def _resolve_bb_columns(self, df: pd.DataFrame):
        bbl_col = next((c for c in df.columns if c.lower().startswith("bbl")), None)
        bbu_col = next((c for c in df.columns if c.lower().startswith("bbu")), None)
        rsi_col = next((c for c in df.columns if c.lower() == "rsi"), None)
        if bbl_col is None or bbu_col is None or rsi_col is None:
            raise KeyError(
                f"Missing required indicator columns. Found columns: {df.columns.tolist()}"
            )
        return bbl_col, bbu_col, rsi_col

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        df = self.apply_indicators(df)
        bbl_col, bbu_col, rsi_col = self._resolve_bb_columns(df)

        signals = pd.Series("HOLD", index=df.index)
        close = pd.to_numeric(df["close"], errors="coerce")
        bbl = pd.to_numeric(df[bbl_col], errors="coerce")
        bbu = pd.to_numeric(df[bbu_col], errors="coerce")
        rsi = pd.to_numeric(df[rsi_col], errors="coerce")

        buy = (close < bbl) & (rsi < 35)
        sell = (close > bbu) & (rsi > 65)
        buy = buy.fillna(False)
        sell = sell.fillna(False)

        signals.loc[buy] = "BUY"
        signals.loc[sell] = "SELL"
        return signals

    def generate_signal(self, df: pd.DataFrame) -> str:
        df = self.apply_indicators(df)
        latest = df.iloc[-1]
        bbl_col, bbu_col, rsi_col = self._resolve_bb_columns(df)

        if latest["close"] < latest[bbl_col] and latest[rsi_col] < 35:
            return "BUY"
        elif latest["close"] > latest[bbu_col] and latest[rsi_col] > 65:
            return "SELL"
        else:
            return "HOLD"


strategy = BollingerRSI()
