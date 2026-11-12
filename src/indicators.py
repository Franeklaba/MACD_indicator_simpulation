"""Wskaźniki analizy technicznej: EMA, MACD (z linią SIGNAL i histogramem), RSI.

Wszystkie wskaźniki są liczone od pierwszej próbki (i = 0) z zachowaniem
pełnej historii, zgodnie ze specyfikacją projektu.
"""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class MacdParams:
    """Parametry konfiguracyjne wskaźnika MACD(N1, N2, N3)."""

    fast: int = 12
    slow: int = 26
    signal: int = 9

    def __post_init__(self) -> None:
        for name in ("fast", "slow", "signal"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(
                    f"Parametr {name} musi być dodatnią liczbą całkowitą "
                    f"(otrzymano: {value!r})."
                )
        if self.fast >= self.slow:
            raise ValueError(
                f"Wymagane N1 < N2 (otrzymano N1={self.fast}, N2={self.slow})."
            )

    def label(self) -> str:
        """Zwraca zapis konfiguracji w postaci MACD(N1, N2, N3)."""
        return f"MACD({self.fast}, {self.slow}, {self.signal})"


@dataclass(frozen=True)
class MacdResult:
    """Wynik obliczeń MACD: linie M_i, S_i oraz histogram H_i."""

    macd: pd.Series
    signal: pd.Series
    histogram: pd.Series
    params: MacdParams


def _check_series(prices: pd.Series) -> None:
    """Sprawdza, czy szereg cen nadaje się do obliczeń wskaźników."""
    if not isinstance(prices, pd.Series):
        raise TypeError("Ceny muszą być przekazane jako obiekt pandas.Series.")
    if prices.empty:
        raise ValueError("Szereg cen jest pusty.")
    if prices.isna().any():
        raise ValueError("Szereg cen zawiera brakujące wartości (NaN).")


def ema(values: pd.Series, period: int) -> pd.Series:
    """Wykładnicza średnia krocząca o okresie N.

    alpha_N = 2 / (N + 1), E_0 = x_0, E_i = alpha_N * x_i + (1 - alpha_N) * E_{i-1}.
    Parametr ``adjust=False`` zapewnia rekurencję zgodną z powyższym wzorem.
    """
    _check_series(values)
    if not isinstance(period, int) or period < 1:
        raise ValueError(f"Okres EMA musi być dodatnią liczbą całkowitą: {period!r}")
    return values.ewm(span=period, adjust=False).mean()


def compute_macd(prices: pd.Series, params: MacdParams | None = None) -> MacdResult:
    """Oblicza linię MACD, linię SIGNAL i histogram dla podanych cen zamknięcia.

    M_i = E_i^(N1) - E_i^(N2),  S_i = EMA_N3(M)_i,  H_i = M_i - S_i.
    """
    params = params or MacdParams()
    macd_line = ema(prices, params.fast) - ema(prices, params.slow)
    signal_line = ema(macd_line, params.signal)
    histogram = macd_line - signal_line
    return MacdResult(
        macd=macd_line.rename("macd"),
        signal=signal_line.rename("signal"),
        histogram=histogram.rename("histogram"),
        params=params,
    )


def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Wskaźnik siły względnej RSI (wygładzanie Wildera, alpha = 1 / N).

    RSI_i = 100 - 100 / (1 + RS_i),  RS_i = avg_gain_i / avg_loss_i.
    Dla i = 0 (brak zmiany ceny) przyjmuje się neutralną wartość 50.
    """
    _check_series(prices)
    if not isinstance(period, int) or period < 1:
        raise ValueError(f"Okres RSI musi być dodatnią liczbą całkowitą: {period!r}")

    delta = prices.diff().fillna(0.0)
    gains = delta.clip(lower=0.0)
    losses = (-delta).clip(lower=0.0)
    avg_gain = gains.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1.0 / period, adjust=False).mean()

    total = avg_gain + avg_loss
    # RSI = 100 * avg_gain / (avg_gain + avg_loss) – postać odporna na avg_loss = 0.
    rsi = (100.0 * avg_gain / total.where(total > 0)).fillna(50.0)
    return rsi.rename("rsi")

