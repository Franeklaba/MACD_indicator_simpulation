"""Testy jednostkowe symulatora portfela i statystyk końcowych."""

import unittest

import numpy as np
import pandas as pd

from src.backtester import BUY, NO_SIGNAL, SELL, simulate
from src.metrics import summarize


class TestBacktester(unittest.TestCase):
    """Weryfikacja reguł portfela C_100 = 1000 * x_100, d in {0, 1} i statystyk."""

    def test_initial_state_and_delay_one(self) -> None:
        """Transakcja dla d = 1 jest wykonywana w próbce i + 1 po cenie x_{i+1}."""
        prices = pd.Series(np.full(120, 10.0))
        prices.iloc[102] = 10.0  # cena wykonania kupna (sygnał w 101)
        prices.iloc[106] = 12.0  # cena wykonania sprzedaży (sygnał w 105)
        prices.iloc[-1] = 12.0

        signals = pd.Series(np.full(120, NO_SIGNAL, dtype=int))
        signals.iloc[101] = BUY
        signals.iloc[105] = SELL

        res = simulate(prices, signals, delay=1)
        self.assertTrue(np.isnan(res.equity[:100]).all())
        self.assertAlmostEqual(res.initial_capital, 10_000.0)
        self.assertEqual(len(res.closed_trades), 1)

        closed = res.closed_trades[0]
        self.assertEqual(closed.buy.signal_index, 101)
        self.assertEqual(closed.buy.exec_index, 102)
        self.assertEqual(closed.sell.signal_index, 105)
        self.assertEqual(closed.sell.exec_index, 106)
        self.assertAlmostEqual(closed.return_pct, 20.0)
        self.assertAlmostEqual(res.total_return_pct, 20.0)

    def test_unexecuted_sell_without_position_and_last_bar_signal(self) -> None:
        """Sprzedaż przy Q = 0 oraz sygnał w próbce L-1 dla d = 1 są niewykonane."""
        prices = pd.Series(np.linspace(10.0, 20.0, 120))
        signals = pd.Series(np.full(120, NO_SIGNAL, dtype=int))
        signals.iloc[101] = SELL  # brak pozycji -> niewykonany
        signals.iloc[119] = BUY   # ostatnia próbka przy d=1 -> niewykonany

        res = simulate(prices, signals, delay=1)
        self.assertEqual(len(res.trades), 0)
        self.assertEqual(len(res.executed_signals), 0)
        self.assertAlmostEqual(res.total_return_pct, 0.0)

    def test_summary_reports_max_loss_as_positive(self) -> None:
        """Największa strata r = -8% jest raportowana jako dodatnia wartość 8%."""
        prices = pd.Series(np.full(120, 100.0))
        prices.iloc[102] = 100.0
        prices.iloc[104] = 92.0  # -8%

        signals = pd.Series(np.full(120, NO_SIGNAL, dtype=int))
        signals.iloc[101] = BUY
        signals.iloc[103] = SELL

        stats = summarize(simulate(prices, signals, delay=1))
        self.assertEqual(stats.n_closed, 1)
        self.assertEqual(stats.n_losing, 1)
        self.assertEqual(stats.n_profitable, 0)
        self.assertAlmostEqual(stats.max_loss_pct or 0.0, 8.0)
        self.assertIsNone(stats.max_gain_pct)


if __name__ == "__main__":
    unittest.main()

