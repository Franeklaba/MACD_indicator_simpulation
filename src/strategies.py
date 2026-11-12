"""Strategie decyzyjne generujące sygnały kupna (+1) i sprzedaży (-1).

* ``MacdStrategy`` – algorytm podstawowy: zmiana znaku histogramu H_i.
* ``FilteredMacdStrategy`` – algorytm zmodyfikowany: sygnały MACD
  przepuszczane przez dodatkowe filtry (RSI, trend EMA, potwierdzenie).
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.backtester import BUY, FIRST_SIGNAL_INDEX, NO_SIGNAL, SELL
from src.indicators import MacdParams, MacdResult, compute_macd, compute_rsi, ema


@dataclass
class StrategyOutput:
    """Sygnały strategii wraz z obliczonymi wskaźnikami (do wykresów)."""

    signals: pd.Series
    macd: MacdResult
    extras: dict[str, pd.Series] = field(default_factory=dict)
    raw_signals: pd.Series | None = None


def crossover_signals(
    histogram: pd.Series, first_index: int = FIRST_SIGNAL_INDEX
) -> pd.Series:
    """Wyznacza sygnały ze zmiany znaku histogramu (od próbki ``first_index``).

    Kupno w i:   H_{i-1} <= 0 oraz H_i > 0.
    Sprzedaż w i: H_{i-1} >= 0 oraz H_i < 0.
    H_i = 0 – brak nowego sygnału.
    """
    h = histogram.to_numpy(dtype=float)
    prev = np.roll(h, 1)
    signals = np.full(len(h), NO_SIGNAL, dtype=int)
    signals[(prev <= 0) & (h > 0)] = BUY
    signals[(prev >= 0) & (h < 0)] = SELL
    signals[:max(first_index, 1)] = NO_SIGNAL
    return pd.Series(signals, index=histogram.index, name="signal")


@dataclass(frozen=True)
class MacdStrategy:
    """Algorytm podstawowy: sygnały z przecięć MACD i SIGNAL."""

    params: MacdParams = MacdParams()
    name: str = "MACD"

    def generate(self, prices: pd.Series) -> StrategyOutput:
        """Oblicza MACD i sygnały dla podanych cen zamknięcia."""
        macd = compute_macd(prices, self.params)
        signals = crossover_signals(macd.histogram)
        return StrategyOutput(signals=signals, macd=macd, raw_signals=signals)


@dataclass(frozen=True)
class FilteredMacdStrategy:
    """Algorytm zmodyfikowany: MACD + filtr RSI i/lub filtr trendu.

    Reguły (stosowane do sygnałów kupna; sygnały sprzedaży bez zmian,
    aby nie wydłużać ekspozycji na spadki):
      * ``rsi_max`` – kupno odrzucane, gdy RSI_i >= rsi_max (rynek
        „wykupiony”, ryzyko zakupu na szczycie),
      * ``trend_period`` – kupno tylko, gdy x_i > EMA_trend_i
        (handel zgodnie z trendem długoterminowym),
      * ``trend_exit`` – dodatkowa sprzedaż, gdy cena przebije EMA trendu
        od góry (ochrona kapitału w bessie).
    """

    params: MacdParams = MacdParams()
    rsi_period: int = 14
    rsi_max: float | None = None
    trend_period: int | None = None
    trend_exit: bool = False
    name: str = "MACD + filtr"

    def generate(self, prices: pd.Series) -> StrategyOutput:
        """Oblicza sygnały MACD i filtruje je regułami modyfikacji."""
        macd = compute_macd(prices, self.params)
        raw = crossover_signals(macd.histogram)
        signals = raw.to_numpy().copy()
        extras: dict[str, pd.Series] = {}
        x = prices.to_numpy(dtype=float)

        buy_ok = np.ones(len(x), dtype=bool)
        if self.rsi_max is not None:
            rsi = compute_rsi(prices, self.rsi_period)
            extras["rsi"] = rsi
            buy_ok &= rsi.to_numpy() < self.rsi_max
        if self.trend_period is not None:
            trend = ema(prices, self.trend_period)
            extras["trend"] = trend.rename("trend")
            above = x > trend.to_numpy()
            buy_ok &= above
            if self.trend_exit:
                below_now = ~above
                cross_down = below_now & np.roll(above, 1)
                cross_down[:FIRST_SIGNAL_INDEX] = False
                signals[cross_down & (signals == NO_SIGNAL)] = SELL

        signals[(signals == BUY) & ~buy_ok] = NO_SIGNAL
        return StrategyOutput(
            signals=pd.Series(signals, index=prices.index, name="signal"),
            macd=macd,
            extras=extras,
            raw_signals=raw,
        )

