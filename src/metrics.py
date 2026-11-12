"""Statystyki końcowe symulacji (podsumowanie liczbowe – slajd F)."""

from dataclasses import asdict, dataclass

from src.backtester import SimulationResult


@dataclass(frozen=True)
class SummaryStats:
    """Podsumowanie liczbowe symulacji zgodne z punktem 4.4 specyfikacji."""

    initial_capital: float
    final_equity: float
    profit: float
    total_return_pct: float
    n_closed: int
    n_profitable: int
    n_losing: int
    win_rate_pct: float | None
    max_loss_pct: float | None
    max_gain_pct: float | None
    mean_trade_return_pct: float | None
    n_signals: int
    n_executed: int
    open_position: bool
    open_position_buy_index: int | None
    open_position_buy_price: float | None

    def to_dict(self) -> dict:
        """Zwraca statystyki w postaci słownika (eksport do JSON/CSV)."""
        return asdict(self)


def summarize(result: SimulationResult) -> SummaryStats:
    """Wyznacza statystyki końcowe na podstawie wyniku symulacji.

    Transakcje z r >= 0 są zyskowne, z r < 0 – stratne. Największa strata
    jest raportowana jako wartość dodatnia; ``None`` oznacza „nie dotyczy”.
    """
    returns = [trade.return_pct for trade in result.closed_trades]
    profitable = [r for r in returns if r >= 0]
    losing = [r for r in returns if r < 0]
    open_buy = result.open_position
    n_signals = int((result.signals != 0).sum())

    return SummaryStats(
        initial_capital=result.initial_capital,
        final_equity=result.final_equity,
        profit=result.final_equity - result.initial_capital,
        total_return_pct=result.total_return_pct,
        n_closed=len(returns),
        n_profitable=len(profitable),
        n_losing=len(losing),
        win_rate_pct=(100.0 * len(profitable) / len(returns)) if returns else None,
        max_loss_pct=-min(losing) if losing else None,
        max_gain_pct=max(profitable) if profitable else None,
        mean_trade_return_pct=(sum(returns) / len(returns)) if returns else None,
        n_signals=n_signals,
        n_executed=len(result.executed_signals),
        open_position=open_buy is not None,
        open_position_buy_index=open_buy.exec_index if open_buy else None,
        open_position_buy_price=open_buy.price if open_buy else None,
    )


def format_summary(stats: SummaryStats, currency: str = "USD") -> str:
    """Formatuje podsumowanie do czytelnego raportu tekstowego (konsola/plik)."""

    def pct(value: float | None) -> str:
        return "nie dotyczy" if value is None else f"{value:.2f}%"

    lines = [
        f"Kapitał początkowy C_100:        {stats.initial_capital:,.2f} {currency}",
        f"Końcowa wartość portfela V_L-1:  {stats.final_equity:,.2f} {currency}",
        f"Zysk/strata kwotowo:             {stats.profit:+,.2f} {currency}",
        f"Stopa zwrotu portfela R:         {stats.total_return_pct:+.2f}%",
        f"Zamknięte transakcje:            {stats.n_closed}",
        f"  zyskowne (r >= 0):             {stats.n_profitable}",
        f"  stratne (r < 0):               {stats.n_losing}",
        f"Odsetek transakcji zyskownych:   {pct(stats.win_rate_pct)}",
        f"Największa strata:               {pct(stats.max_loss_pct)}",
        f"Największy zysk:                 {pct(stats.max_gain_pct)}",
        f"Sygnały / wykonane:              {stats.n_signals} / {stats.n_executed}",
    ]
    if stats.open_position:
        lines.append(
            "Pozycja otwarta na końcu:        tak (kupno w próbce "
            f"{stats.open_position_buy_index} po {stats.open_position_buy_price:.2f} "
            f"{currency})"
        )
    else:
        lines.append("Pozycja otwarta na końcu:        nie")
    return "\n".join(lines)

