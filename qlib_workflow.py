import qlib
from qlib.data import D
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.strategy import TopkDropoutStrategy
from qlib.contrib.evaluate import backtest

# 1. Initialize Qlib (use default local data provider)
qlib.init(provider_uri="~/.qlib/qlib_data/cn_data")  # or use "market=us" for NASDAQ

# 2. Fetch data for a single stock
df = D.features(
    instruments=["AAPL"],
    fields=["$close", "$open", "$high", "$low", "$volume"],
    start_time="2023-01-01",
    end_time="2023-09-01"
)
print(df.head())

# 3. Load a pre-trained model (LightGBM example)
model = LGBModel.load_from_file("path/to/pretrained_model.pkl")
predictions = model.predict(df)
print(predictions.head())

# 4. Backtest a strategy
strategy = TopkDropoutStrategy(topk=5)
report = backtest(predictions, strategy, initial_cash=100000)
print(report)
