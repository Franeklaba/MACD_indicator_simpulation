"""Część eksperymentalna: przetwarzanie wsadowe i porównanie algorytmów.

Dla każdej spółki uruchamiane są niezależnie: algorytm podstawowy MACD,
algorytm zmodyfikowany (MACD + filtr RSI) oraz strategia „kup i trzymaj”.
Różnice stóp zwrotu wyrażane są w punktach procentowych (p.p.).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.backtester import (
    BUY,
    SimulationResult,
    buy_and_hold_return_pct,
    simulate,
)
from src.data_loader import StockData
from src.metrics import summarize

EPS = 1e-9


@dataclass(frozen=True)
class BlockedTrade:
    """Transakcja bazowa, której kupno zostało zablokowane przez filtr."""

    ticker: str
    signal_index: int
    buy_index: int
    sell_index: int | None
    buy_price: float
    sell_price: float | None
    return_pct: float | None
    rsi_at_signal: float


@dataclass
class ExperimentResult:
    """Wyniki części eksperymentalnej dla całego zbioru spółek."""

    table: pd.DataFrame
    blocked: pd.DataFrame
    sensitivity: pd.DataFrame
    kept_returns: pd.Series
    def global_summary(self) -> dict[str, float]:
        """Statystyki globalne: średnia, mediana i liczności zmian ΔR."""
        diff = self.table["delta_pp"]
        return {
            "n_tickers": int(len(diff)),
            "mean_delta_pp": float(diff.mean()),
            "median_delta_pp": float(diff.median()),
            "n_better": int((diff > EPS).sum()),
            "n_worse": int((diff < -EPS).sum()),
            "n_same": int((diff.abs() <= EPS).sum()),
            "mean_r_base": float(self.table["r_base"].mean()),
            "median_r_base": float(self.table["r_base"].median()),
            "mean_r_mod": float(self.table["r_mod"].mean()),
            "median_r_mod": float(self.table["r_mod"].median()),
            "mean_r_bh": float(self.table["r_bh"].mean()),
            "median_r_bh": float(self.table["r_bh"].median()),
            "base_beats_bh": int((self.table["r_base"] > self.table["r_bh"]).sum()),
            "mod_beats_bh": int((self.table["r_mod"] > self.table["r_bh"]).sum()),
        }


def _row(
    ticker: str, base: SimulationResult, mod: SimulationResult, r_bh: float
) -> dict:
    """Buduje wiersz tabeli porównawczej dla jednej spółki."""
    s_base, s_mod = summarize(base), summarize(mod)
    return {
        "ticker": ticker,
        "r_base": s_base.total_return_pct,
        "r_mod": s_mod.total_return_pct,
        "r_bh": r_bh,
        "delta_pp": s_mod.total_return_pct - s_base.total_return_pct,
        "trades_base": s_base.n_closed,
        "trades_mod": s_mod.n_closed,
        "win_base": s_base.win_rate_pct,
        "win_mod": s_mod.win_rate_pct,
        "max_loss_base": s_base.max_loss_pct or 0.0,
        "max_loss_mod": s_mod.max_loss_pct or 0.0,
    }


def find_blocked_trades(
    ticker: str,
    base: SimulationResult,
    mod: SimulationResult,
    rsi: pd.Series,
) -> list[BlockedTrade]:
    """Identyfikuje transakcje bazowe pominięte przez algorytm zmodyfikowany."""
    mod_buys = {t.signal_index for t in mod.trades if t.side == BUY}
    sells_by_buy = {c.buy.signal_index: c for c in base.closed_trades}
    blocked: list[BlockedTrade] = []
    for trade in base.trades:
        if trade.side != BUY or trade.signal_index in mod_buys:
            continue
        closed = sells_by_buy.get(trade.signal_index)
        blocked.append(
            BlockedTrade(
                ticker=ticker,
                signal_index=trade.signal_index,
                buy_index=trade.exec_index,
                sell_index=closed.sell.exec_index if closed else None,
                buy_price=trade.price,
                sell_price=closed.sell.price if closed else None,
                return_pct=closed.return_pct if closed else None,
                rsi_at_signal=float(rsi.iloc[trade.signal_index]),
            )
        )
    return blocked


def run_experiment(
    stocks: dict[str, StockData],
    base_strategy,
    mod_strategy,
    delay: int = 1,
    sensitivity_strategies: dict[str, object] | None = None,
) -> ExperimentResult:
    """Uruchamia porównanie algorytmów na wszystkich spółkach ze zbioru.

    Args:
        stocks: słownik {ticker: StockData}.
        base_strategy: algorytm podstawowy (obiekt z metodą ``generate``).
        mod_strategy: algorytm zmodyfikowany (musi wyznaczać ``extras['rsi']``).
        delay: opóźnienie wykonania d.
        sensitivity_strategies: dodatkowe warianty do analizy wrażliwości.
    """
    if not stocks:
        raise ValueError("Brak danych wejściowych do części eksperymentalnej.")

    rows: list[dict] = []
    blocked: list[BlockedTrade] = []
    kept: list[float] = []
    for ticker, stock in stocks.items():
        prices = stock.df["close"]
        base_out = base_strategy.generate(prices)
        mod_out = mod_strategy.generate(prices)
        base = simulate(prices, base_out.signals, delay=delay)
        mod = simulate(prices, mod_out.signals, delay=delay)
        rows.append(_row(ticker, base, mod, buy_and_hold_return_pct(prices)))
        kept.extend(trade.return_pct for trade in mod.closed_trades)
        if "rsi" in mod_out.extras:
            blocked.extend(
                find_blocked_trades(ticker, base, mod, mod_out.extras["rsi"])
            )

    table = pd.DataFrame(rows).set_index("ticker")
    blocked_df = pd.DataFrame([b.__dict__ for b in blocked])
    sensitivity = _sensitivity(stocks, table, sensitivity_strategies or {}, delay)
    return ExperimentResult(
        table=table,
        blocked=blocked_df,
        sensitivity=sensitivity,
        kept_returns=pd.Series(kept, name="return_pct"),
    )


def _sensitivity(
    stocks: dict[str, StockData],
    table: pd.DataFrame,
    strategies: dict[str, object],
    delay: int,
) -> pd.DataFrame:
    """Analiza wrażliwości: wyniki globalne dla alternatywnych wariantów."""
    records = []
    r_base = table["r_base"]
    for label, strategy in strategies.items():
        returns, trades, wins = [], [], []
        for ticker, stock in stocks.items():
            prices = stock.df["close"]
            result = simulate(prices, strategy.generate(prices).signals, delay)
            stats = summarize(result)
            returns.append(stats.total_return_pct)
            trades.append(stats.n_closed)
            if stats.win_rate_pct is not None:
                wins.append(stats.win_rate_pct)
        r = pd.Series(returns, index=list(stocks))
        diff = r - r_base.loc[r.index]
        records.append({
            "wariant": label,
            "mean_r": r.mean(),
            "median_r": r.median(),
            "mean_delta_pp": diff.mean(),
            "median_delta_pp": diff.median(),
            "n_better": int((diff > EPS).sum()),
            "n_worse": int((diff < -EPS).sum()),
            "n_same": int((diff.abs() <= EPS).sum()),
            "mean_trades": float(np.mean(trades)),
            "mean_win_pct": float(np.mean(wins)) if wins else np.nan,
        })
    return pd.DataFrame(records)
