import warnings

import numpy as np
import pandas as pd

from ai.pipeline import TimeSeriesPipeline


def test_pipeline_predict_preserves_feature_names():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(120, 6)), columns=[f"f{i}" for i in range(6)])
    y = 0.3 * X["f0"] - 0.2 * X["f1"] + rng.normal(scale=0.05, size=len(X))

    pipeline = TimeSeriesPipeline(n_features=3, verbose=False)
    pipeline.train_final_model(X, y, test_size=0.2)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipeline.predict(X.iloc[:10])

    messages = [str(w.message) for w in caught]
    assert not any("does not have valid feature names" in msg for msg in messages)
