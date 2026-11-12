"""Silnik symulacji portfela sterowanego sygnałami kupna i sprzedaży.

Konwencja sygnałów: +1 – kupno, -1 – sprzedaż, 0 – brak sygnału.
Portfel startuje w próbce ``start`` (domyślnie 100) z gotówką C = 1000 * x_start,
a transakcja na podstawie sygnału z próbki i - d jest wykonywana w próbce i
po cenie x_i (d = 1 – wariant zalecany, d = 0 – wariant uproszczony).
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

BUY = 1
SELL = -1
NO_SIGNAL = 0

START_INDEX = 100
FIRST_SIGNAL_INDEX = START_INDEX + 1
CAPITAL_UNITS = 1000.0


@dataclass(frozen=True)
class Trade:
    """Pojedyncza wykonana operacja (kupno lub sprzedaż)."""

    side: int
    signal_index: int
    exec_index: int
    price: float
    quantity: float


@dataclass(frozen=True)
class ClosedTrade:
    """Zamknięta transakcja: kupno w próbce a i sprzedaż w próbce b."""

    buy: Trade
    sell: Trade

    @property
    def return_pct(self) -> float:
        """Stopa zwrotu r = 100 * (x_b / x_a - 1) [%]."""
        return 100.0 * (self.sell.price / self.buy.price - 1.0)


@dataclass
class SimulationResult:
    """Pełny zapis przebiegu symulacji (wektory C, Q, V oraz transakcje)."""

    cash: np.ndarray
    position_qty: np.ndarray
    equity: np.ndarray
    trades: list[Trade]
    signals: pd.Series
    executed_signals: set[int] = field(default_factory=set)
    start: int = START_INDEX
    delay: int = 1

    @property
    def initial_capital(self) -> float:
        """Kapitał początkowy C_start."""
        return float(self.cash[self.start])

    @property
    def final_equity(self) -> float:
        """Końcowa wartość portfela V_{L-1}."""
        return float(self.equity[-1])

    @property
    def total_return_pct(self) -> float:
        """Stopa zwrotu portfela R = 100 * (V_{L-1} / C_start - 1) [%]."""
        return 100.0 * (self.final_equity / self.initial_capital - 1.0)

    @property
    def closed_trades(self) -> list[ClosedTrade]:
        """Pary kupno–sprzedaż (kolejność operacji jest zawsze naprzemienna)."""
        pairs: list[ClosedTrade] = []
        open_buy: Trade | None = None
        for trade in self.trades:
            if trade.side == BUY:
                open_buy = trade
            elif open_buy is not None:
                pairs.append(ClosedTrade(buy=open_buy, sell=trade))
                open_buy = None
        return pairs

    @property
    def open_position(self) -> Trade | None:
        """Ostatnie kupno, jeśli pozycja pozostała otwarta na końcu symulacji."""
        if self.trades and self.trades[-1].side == BUY:
            return self.trades[-1]
        return None

    @property
    def signal_indices(self) -> dict[int, list[int]]:
        """Indeksy wszystkich sygnałów kupna i sprzedaży."""
        values = self.signals.to_numpy()
        return {
            BUY: np.flatnonzero(values == BUY).tolist(),
            SELL: np.flatnonzero(values == SELL).tolist(),
        }


def simulate(
    prices: pd.Series,
    signals: pd.Series,
    delay: int = 1,
    start: int = START_INDEX,
) -> SimulationResult:
    """Symuluje portfel typu „all-in / all-out” bez prowizji i krótkiej sprzedaży.

    Args:
        prices: ceny zamknięcia x_0, ..., x_{L-1}.
        signals: sygnały (+1/-1/0) tej samej długości co ``prices``.
        delay: opóźnienie wykonania d (0 lub 1).
        start: indeks próbki stanu początkowego portfela (domyślnie 100).

    Returns:
        Obiekt ``SimulationResult`` z wektorami C, Q, V i listą transakcji.
    """
    if delay not in (0, 1):
        raise ValueError(f"Opóźnienie d musi wynosić 0 lub 1 (otrzymano {delay}).")
    if len(prices) != len(signals):
        raise ValueError("Wektory cen i sygnałów muszą mieć tę samą długość.")
    if not 0 <= start < len(prices) - 1:
        raise ValueError(
            f"Indeks startu {start} poza zakresem dla L = {len(prices)}."
        )

    x = prices.to_numpy(dtype=float)
    sig = signals.to_numpy(dtype=int)
    length = len(x)
    first_signal = start + 1

    cash = np.full(length, np.nan)
    qty = np.full(length, np.nan)
    equity = np.full(length, np.nan)
    cash[start] = CAPITAL_UNITS * x[start]
    qty[start] = 0.0
    equity[start] = cash[start]

    trades: list[Trade] = []
    executed: set[int] = set()

    for i in range(start + 1, length):
        cash[i], qty[i] = cash[i - 1], qty[i - 1]
        source = i - delay
        if source >= first_signal:
            if sig[source] == BUY and cash[i - 1] > 0:
                qty[i] = cash[i - 1] / x[i]
                cash[i] = 0.0
                trades.append(Trade(BUY, source, i, x[i], qty[i]))
                executed.add(source)
            elif sig[source] == SELL and qty[i - 1] > 0:
                cash[i] = qty[i - 1] * x[i]
                trades.append(Trade(SELL, source, i, x[i], qty[i - 1]))
                qty[i] = 0.0
                executed.add(source)
        equity[i] = cash[i] + qty[i] * x[i]

    return SimulationResult(
        cash=cash,
        position_qty=qty,
        equity=equity,
        trades=trades,
        signals=signals,
        executed_signals=executed,
        start=start,
        delay=delay,
    )


def buy_and_hold_return_pct(prices: pd.Series, start: int = START_INDEX) -> float:
    """Stopa zwrotu strategii „kup i trzymaj” (zakup w próbce start) [%]."""
    x = prices.to_numpy(dtype=float)
    return 100.0 * (x[-1] / x[start] - 1.0)

