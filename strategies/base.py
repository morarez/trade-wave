from __future__ import annotations

import abc
from typing import Any

import pandas as pd


class Strategy(abc.ABC):
    """Base strategy interface for historical and live signal generation."""

    name: str = "unnamed"

    @abc.abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate signals for the full historical input DataFrame."""
        raise NotImplementedError

    @abc.abstractmethod
    def generate_signal(self, df: pd.DataFrame) -> str:
        """Generate a single signal for the latest row of the input DataFrame."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<Strategy {self.name}>"
