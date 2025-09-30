import requests
import pandas as pd
import os

def print_preview(df, n=5):
    """Helper to preview DataFrames in a consistent format"""
    print(df.head(n))
    print(f"\nShape: {df.shape}")

def save_signal(symbol, signal, filename="data/signals.csv"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df_new = pd.DataFrame([{"symbol": symbol, "signal": signal}])
    
    if os.path.exists(filename):
        df_new.to_csv(filename, mode='a', header=False, index=False)
    else:
        df_new.to_csv(filename, index=False)

