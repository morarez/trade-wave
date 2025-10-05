from strategies import sma_rsi, bollinger_rsi, macd_trend

STRATEGY_MAP = {
    "sma_rsi": sma_rsi,
    "bollinger_rsi": bollinger_rsi,
    "macd_trend": macd_trend
}

def get_strategy(name: str):
    return STRATEGY_MAP.get(name)
