from strategies.sma_rsi import strategy as sma_rsi
from strategies.bollinger_rsi import strategy as bollinger_rsi
from strategies.macd_trend import strategy as macd_trend
from strategies.ai_strategy import strategy as ai_strategy

STRATEGY_MAP = {
    "sma_rsi": sma_rsi,
    "bollinger_rsi": bollinger_rsi,
    "macd_trend": macd_trend,
    "ai_model": ai_strategy,
}

def get_strategy(name: str):
    """Retrieve a strategy module by name.
    
    Args:
        name: strategy name (e.g., 'sma_rsi', 'bollinger_rsi', 'macd_trend').
    
    Returns:
        Strategy module or None if not found.
    """
    return STRATEGY_MAP.get(name)
