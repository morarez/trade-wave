import pandas as pd
import pytest

from api import serialize_dataframe


def test_serialize_dataframe_raises_for_non_finite_values():
    df = pd.DataFrame({"metric": [1.0, float("inf"), -float("inf"), None]})

    with pytest.raises(ValueError, match="non-finite|NaN"):
        serialize_dataframe(df)
