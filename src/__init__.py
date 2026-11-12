"""Pakiet źródłowy symulatora wskaźnika MACD (Projekt 1)."""

from src.backtester import (
    BUY,
    NO_SIGNAL,
    SELL,
    ClosedTrade,
    SimulationResult,
    Trade,
    buy_and_hold_return_pct,
    simulate,
)
from src.data_loader import DataLoader, StockData
from src.experiment import ExperimentResult, run_experiment
from src.indicators import MacdParams, MacdResult, compute_macd, compute_rsi, ema
from src.metrics import SummaryStats, format_summary, summarize
from src.strategies import (
    FilteredMacdStrategy,
    MacdStrategy,
    StrategyOutput,
    crossover_signals,
)

__all__ = [
    "BUY",
    "NO_SIGNAL",
    "SELL",
    "ClosedTrade",
    "DataLoader",
    "ExperimentResult",
    "FilteredMacdStrategy",
    "MacdParams",
    "MacdResult",
    "MacdStrategy",
    "SimulationResult",
    "StockData",
    "StrategyOutput",
    "SummaryStats",
    "Trade",
    "buy_and_hold_return_pct",
    "compute_macd",
    "compute_rsi",
    "crossover_signals",
    "ema",
    "format_summary",
    "run_experiment",
    "simulate",
    "summarize",
]
