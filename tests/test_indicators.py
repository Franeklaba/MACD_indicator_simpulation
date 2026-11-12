"""Testy jednostkowe modułu wskaźników (EMA, MACD, RSI) oraz sygnałów."""

import unittest

import numpy as np
import pandas as pd

from src.backtester import BUY, NO_SIGNAL, SELL
from src.indicators import MacdParams, compute_macd, compute_rsi, ema
from src.strategies import crossover_signals


class TestIndicators(unittest.TestCase):
    """Weryfikacja wzorów rekurencyjnych EMA, MACD, SIGNAL, H i RSI."""

    def test_ema_matches_explicit_recurrence(self) -> None:
        """EMA obliczone przez pandas.ewm(adjust=False) zgadza się ze wzorem (1)."""
        prices = pd.Series([100.0, 102.0, 101.0, 105.0, 103.0, 108.0])
        period = 12
        alpha = 2.0 / (period + 1)
        expected = [prices.iloc[0]]
        for x in prices.iloc[1:]:
            expected.append(alpha * x + (1.0 - alpha) * expected[-1])
        result = ema(prices, period)
        np.testing.assert_allclose(result.to_numpy(), expected, rtol=1e-12)

    def test_macd_initial_values_zero(self) -> None:
        """W próbce i = 0 zachodzi E_0^(N1) = E_0^(N2) = x_0, więc M_0 = S_0 = 0."""
        prices = pd.Series(np.linspace(50.0, 150.0, 200))
        res = compute_macd(prices)
        self.assertAlmostEqual(res.macd.iloc[0], 0.0)
        self.assertAlmostEqual(res.signal.iloc[0], 0.0)
        self.assertAlmostEqual(res.histogram.iloc[0], 0.0)
        np.testing.assert_allclose(
            res.histogram.to_numpy(),
            (res.macd - res.signal).to_numpy(),
            rtol=1e-12,
        )

    def test_macd_invalid_params(self) -> None:
        """N1 >= N2 lub niedodatnie okresy powinny zgłaszać ValueError."""
        with self.assertRaises(ValueError):
            MacdParams(fast=26, slow=12, signal=9)
        with self.assertRaises(ValueError):
            MacdParams(fast=0, slow=26, signal=9)

    def test_crossover_signals_rules_and_warmup(self) -> None:
        """Sygnały przed próbką 101 są ignorowane; H_i = 0 nie generuje sygnału."""
        hist = pd.Series(np.zeros(110))
        hist.iloc[50] = -1.0
        hist.iloc[51] = 1.0  # w okresie rozruchu -> brak sygnału
        hist.iloc[100] = -0.5
        hist.iloc[101] = 0.2  # kupno w próbce 101
        hist.iloc[102] = 0.0  # H_i == 0 -> brak sygnału
        hist.iloc[103] = -0.1  # H_102 == 0 >= 0 oraz H_103 < 0 -> sprzedaż
        sig = crossover_signals(hist, first_index=101)
        self.assertEqual(int((sig.iloc[:101] != NO_SIGNAL).sum()), 0)
        self.assertEqual(sig.iloc[101], BUY)
        self.assertEqual(sig.iloc[102], NO_SIGNAL)
        self.assertEqual(sig.iloc[103], SELL)

    def test_rsi_bounds(self) -> None:
        """RSI przyjmuje wartości z przedziału [0, 100]."""
        rng = np.random.default_rng(42)
        prices = pd.Series(100.0 + np.cumsum(rng.normal(0.0, 1.0, 300)))
        rsi = compute_rsi(prices, period=14)
        self.assertTrue(((rsi >= 0.0) & (rsi <= 100.0)).all())


if __name__ == "__main__":
    unittest.main()
