"""Wizualizacja wyników: wykresy A1, B1, C1/C2, D1, E1/E2 oraz eksperymentu.

Wszystkie wykresy zapisywane są wyłącznie w formacie wektorowym PDF.
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from src.backtester import BUY, SELL, ClosedTrade, SimulationResult  # noqa: E402
from src.indicators import MacdResult  # noqa: E402

VECTOR_FORMATS = {".pdf", ".svg", ".eps"}

COLORS = {
    "price": "#1f3b63",
    "price_bg": "#a9b8cc",
    "macd": "#1565c0",
    "signal": "#ef6c00",
    "buy": "#1b8a2f",
    "sell": "#c62828",
    "neutral": "#7f7f7f",
    "equity": "#283593",
    "bh": "#9e9e9e",
    "warmup": "#eceff1",
    "hold": "#e8f5e9",
    "blocked": "#fff3e0",
    "rsi": "#6a1b9a",
}

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "legend.fontsize": 9.5,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.6,
    "lines.linewidth": 1.5,
    "pdf.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
})


@dataclass(frozen=True)
class Instrument:
    """Opis instrumentu wykorzystywany w tytułach i opisach osi."""

    ticker: str
    name: str
    currency: str = "USD"

    @property
    def label(self) -> str:
        """Etykieta w postaci „Apple Inc. (AAPL)”."""
        return f"{self.name} ({self.ticker})" if self.name else self.ticker


def save_figure(fig: plt.Figure, path: Path) -> Path:
    """Zapisuje wykres wyłącznie w formacie wektorowym i zamyka figurę."""
    path = Path(path)
    if path.suffix.lower() not in VECTOR_FORMATS:
        raise ValueError(
            f"Niedozwolony format {path.suffix!r} – dopuszczalne: "
            f"{sorted(VECTOR_FORMATS)}."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def _date_axis(ax: plt.Axes, fmt: str = "%Y-%m") -> None:
    """Ustawia czytelny format dat na osi X."""
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))
    ax.set_xlabel("Data sesji")


def _fmt_date(value: pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _date_range(dates: pd.Series) -> str:
    return f"{_fmt_date(dates.iloc[0])} – {_fmt_date(dates.iloc[-1])}"


# ----------------------------------------------------------------------------
# A1 – dane wejściowe
# ----------------------------------------------------------------------------
def plot_a1_price(
    df: pd.DataFrame, inst: Instrument, path: Path, warmup_end: int = 100
) -> Path:
    """A1: cena zamknięcia dla wszystkich L próbek z okresem rozruchu 0..100."""
    dates, close = df["date"], df["close"]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.axvspan(dates.iloc[0], dates.iloc[warmup_end], color=COLORS["warmup"],
               label=f"Rozruch średnich (próbki 0–{warmup_end}, bez sygnałów)")
    ax.plot(dates, close, color=COLORS["price"], lw=1.3,
            label="Cena zamknięcia $x_i$")
    ax.axvline(dates.iloc[warmup_end], color=COLORS["neutral"], ls="--", lw=1,
               label=f"Start portfela (próbka {warmup_end}, "
                     f"{_fmt_date(dates.iloc[warmup_end])})")
    ax.set_title(f"A1: Cena zamknięcia {inst.label}, L = {len(df)} sesji, "
                 f"{_date_range(dates)}")
    ax.set_ylabel(f"Cena zamknięcia [{inst.currency}]")
    _date_axis(ax)
    ax.set_xlim(dates.iloc[0], dates.iloc[-1])
    ax.legend(loc="upper left")
    return save_figure(fig, path)


# ----------------------------------------------------------------------------
# B1 – cena i wskaźnik (dwie osie Y)
# ----------------------------------------------------------------------------
def _price_macd_axes(
    ax: plt.Axes,
    dates: pd.Series,
    close: pd.Series,
    macd: MacdResult,
    inst: Instrument,
) -> tuple[plt.Axes, list]:
    """Rysuje cenę (lewa oś, tło) oraz MACD i SIGNAL (prawa oś, pierwszy plan)."""
    ax.plot(dates, close, color=COLORS["price_bg"], lw=1.6, zorder=1,
            label="Cena zamknięcia $x_i$ (lewa oś)")
    ax.set_ylabel(f"Cena zamknięcia [{inst.currency}]",
                  color=COLORS["price"])
    ax.tick_params(axis="y", colors=COLORS["price"])
    ax.grid(False)

    ax_r = ax.twinx()
    ax_r.plot(dates, macd.macd, color=COLORS["macd"], lw=1.3,
              label="MACD $M_i$ (prawa oś)")
    ax_r.plot(dates, macd.signal, color=COLORS["signal"], lw=1.3, ls="--",
              label="SIGNAL $S_i$ (prawa oś)")
    ax_r.axhline(0, color=COLORS["neutral"], lw=0.7, ls=":")
    ax_r.set_ylabel(f"MACD, SIGNAL [{inst.currency}]", color=COLORS["macd"])
    ax_r.tick_params(axis="y", colors=COLORS["macd"])
    ax_r.grid(True, alpha=0.3)
    # Linie wskaźnika na pierwszym planie, cena w tle.
    ax_r.set_zorder(ax.get_zorder() + 1)
    ax_r.patch.set_visible(False)
    handles = ax.get_legend_handles_labels()[0]
    handles += ax_r.get_legend_handles_labels()[0]
    return ax_r, handles


def plot_b1_price_macd(
    df: pd.DataFrame, macd: MacdResult, inst: Instrument, path: Path
) -> Path:
    """B1: cena (lewa oś Y) oraz MACD i SIGNAL (prawa oś Y) na wspólnej osi czasu."""
    fig, ax = plt.subplots(figsize=(10, 4.2))
    _, handles = _price_macd_axes(ax, df["date"], df["close"], macd, inst)
    ax.set_title(f"B1: Cena {inst.label} oraz {macd.params.label()}, "
                 f"{_date_range(df['date'])}")
    _date_axis(ax)
    ax.set_xlim(df["date"].iloc[0], df["date"].iloc[-1])
    ax.legend(handles=handles, loc="upper left", ncol=3)
    return save_figure(fig, path)


# ----------------------------------------------------------------------------
# C1/C2 – sygnały i histogram
# ----------------------------------------------------------------------------
def plot_c_signals_histogram(
    df: pd.DataFrame,
    macd: MacdResult,
    signals: pd.Series,
    inst: Instrument,
    path: Path,
    start: int = 100,
) -> Path:
    """C1: MACD i SIGNAL z sygnałami; C2: histogram H_i względem zera."""
    sl = slice(start, None)
    dates = df["date"].iloc[sl]
    m, s, h = macd.macd.iloc[sl], macd.signal.iloc[sl], macd.histogram.iloc[sl]
    sig = signals.iloc[sl]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 5.6), sharex=True,
        gridspec_kw={"height_ratios": [1.35, 1], "hspace": 0.28},
    )
    ax1.plot(dates, m, color=COLORS["macd"], lw=1.3, label="MACD $M_i$")
    ax1.plot(dates, s, color=COLORS["signal"], lw=1.3, ls="--",
             label="SIGNAL $S_i$")
    ax1.axhline(0, color=COLORS["neutral"], lw=0.7, ls=":")
    buy, sell = sig == BUY, sig == SELL
    ax1.scatter(dates[buy], m[buy], marker="^", s=55, color=COLORS["buy"],
                edgecolor="black", lw=0.5, zorder=5,
                label=f"Sygnał kupna ({int(buy.sum())})")
    ax1.scatter(dates[sell], m[sell], marker="v", s=55, color=COLORS["sell"],
                edgecolor="black", lw=0.5, zorder=5,
                label=f"Sygnał sprzedaży ({int(sell.sum())})")
    ax1.set_title(f"C1: {macd.params.label()} i sygnały – {inst.label} "
                  f"(od próbki {start + 1})")
    ax1.set_ylabel(f"MACD, SIGNAL [{inst.currency}]")
    ax1.legend(loc="upper left", ncol=4)

    width = 1.0
    ax2.bar(dates[h >= 0], h[h >= 0], width=width, color=COLORS["buy"],
            alpha=0.75, label="$H_i > 0$ (MACD nad SIGNAL)")
    ax2.bar(dates[h < 0], h[h < 0], width=width, color=COLORS["sell"],
            alpha=0.75, label="$H_i < 0$ (MACD pod SIGNAL)")
    ax2.axhline(0, color="black", lw=0.8, label="Poziom zero")
    ax2.set_title(f"C2: Histogram $H_i = M_i - S_i$ – {inst.label}")
    ax2.set_ylabel(f"Histogram $H_i$ [{inst.currency}]")
    ax2.legend(loc="upper left", ncol=3)
    _date_axis(ax2)
    ax2.set_xlim(dates.iloc[0], dates.iloc[-1])
    return save_figure(fig, path)


# ----------------------------------------------------------------------------
# D1 – wybrana transakcja
# ----------------------------------------------------------------------------
def plot_d1_trade(
    df: pd.DataFrame,
    macd: MacdResult,
    trade: ClosedTrade,
    inst: Instrument,
    path: Path,
    margin: int = 25,
) -> Path:
    """D1: powiększenie B1 wokół zamkniętej transakcji (sygnały i wykonania)."""
    lo = max(trade.buy.signal_index - margin, 0)
    hi = min(trade.sell.exec_index + margin, len(df) - 1)
    window = slice(lo, hi + 1)
    sub = df.iloc[window]
    sub_macd = MacdResult(
        macd=macd.macd.iloc[window], signal=macd.signal.iloc[window],
        histogram=macd.histogram.iloc[window], params=macd.params,
    )
    dates, close = df["date"], df["close"]

    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax_r, handles = _price_macd_axes(ax, sub["date"], sub["close"],
                                     sub_macd, inst)
    # Cena wyraźniej – w tym wykresie oznaczamy na niej wykonania.
    ax.lines[0].set_color(COLORS["price"])
    ax.lines[0].set_alpha(0.55)
    ax.axvspan(dates.iloc[trade.buy.exec_index],
               dates.iloc[trade.sell.exec_index],
               color=COLORS["hold"], zorder=0, label="Pozycja otwarta")

    for t, color, marker, name in (
        (trade.buy, COLORS["buy"], "^", "kupna"),
        (trade.sell, COLORS["sell"], "v", "sprzedaży"),
    ):
        ax_r.scatter(dates.iloc[t.signal_index],
                     macd.macd.iloc[t.signal_index], marker=marker, s=90,
                     facecolor="white", edgecolor=color, lw=2, zorder=6,
                     label=f"Sygnał {name} (próbka {t.signal_index})")
        ax.scatter(dates.iloc[t.exec_index], t.price, marker=marker, s=110,
                   color=color, edgecolor="black", lw=0.6, zorder=6,
                   label=f"Wykonanie {name} (próbka {t.exec_index})")

    offset = 0.06 * (sub["close"].max() - sub["close"].min())
    for t, sign, name in ((trade.buy, -1, "Kupno"), (trade.sell, 1, "Sprzedaż")):
        ax.annotate(
            f"{name}: {_fmt_date(dates.iloc[t.exec_index])}\n"
            f"{t.price:.2f} {inst.currency}",
            xy=(dates.iloc[t.exec_index], t.price),
            xytext=(dates.iloc[t.exec_index],
                    t.price + sign * 3.2 * offset),
            ha="center", fontsize=9.5,
            bbox={"boxstyle": "round", "fc": "white", "ec": "0.6"},
            arrowprops={"arrowstyle": "->", "color": "0.3"},
            zorder=7,
        )
    ax.text(0.99, 0.04,
            f"r = 100·({trade.sell.price:.2f}/{trade.buy.price:.2f} − 1)"
            f" = {trade.return_pct:+.2f}%",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=10.5,
            fontweight="bold",
            bbox={"boxstyle": "round", "fc": "#fffde7", "ec": "0.5"},
            zorder=8)

    ax.set_title(f"D1: Transakcja {inst.label} "
                 f"{_fmt_date(dates.iloc[trade.buy.exec_index])} – "
                 f"{_fmt_date(dates.iloc[trade.sell.exec_index])} "
                 f"(powiększenie B1, d = 1)")
    _date_axis(ax, "%Y-%m-%d")
    ax.set_xlim(sub["date"].iloc[0], sub["date"].iloc[-1])
    handles_l, _ = ax.get_legend_handles_labels()
    handles_r, _ = ax_r.get_legend_handles_labels()
    ax.legend(handles=handles_l + handles_r, loc="upper left", ncol=3,
              fontsize=8.5)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin - 0.25 * (ymax - ymin), ymax + 0.45 * (ymax - ymin))
    rmin, rmax = ax_r.get_ylim()
    ax_r.set_ylim(rmin - 0.1 * (rmax - rmin), rmax + 0.6 * (rmax - rmin))
    return save_figure(fig, path)


# ----------------------------------------------------------------------------
# E1/E2 – cała symulacja
# ----------------------------------------------------------------------------
def plot_e_simulation(
    df: pd.DataFrame,
    result: SimulationResult,
    inst: Instrument,
    path: Path,
    bh_equity: np.ndarray | None = None,
) -> Path:
    """E1: cena z sygnałami (wykonane/niewykonane); E2: wartość portfela V_i."""
    start = result.start
    dates, close = df["date"], df["close"]
    sl = slice(start, None)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 5.8), sharex=True,
        gridspec_kw={"height_ratios": [1.25, 1], "hspace": 0.28},
    )
    ax1.plot(dates.iloc[sl], close.iloc[sl], color=COLORS["price"], lw=1.1,
             label="Cena zamknięcia $x_i$")

    for side, color, marker, name in (
        (BUY, COLORS["buy"], "^", "kupna"),
        (SELL, COLORS["sell"], "v", "sprzedaży"),
    ):
        idx = result.signal_indices[side]
        done = [i for i in idx if i in result.executed_signals]
        skipped = [i for i in idx if i not in result.executed_signals]
        execs = [t.exec_index for t in result.trades if t.side == side]
        ax1.scatter(dates.iloc[done], close.iloc[done], marker=marker, s=28,
                    facecolor="white", edgecolor=color, lw=1.1, zorder=4,
                    label=f"Sygnał {name} – moment sygnału ({len(done)})")
        ax1.scatter(dates.iloc[execs], close.iloc[execs], marker=marker, s=42,
                    color=color, edgecolor="black", lw=0.4, zorder=5,
                    label=f"Wykonanie {name} (d = {result.delay})")
        ax1.scatter(dates.iloc[skipped], close.iloc[skipped], marker=marker,
                    s=70, facecolor="none", edgecolor="black", lw=1.4,
                    hatch="////", zorder=6,
                    label=f"Sygnał {name} niewykonany ({len(skipped)})")
    ax1.set_title(f"E1: Cena {inst.label} i sygnały MACD(12, 26, 9) – "
                  f"symulacja od próbki {start}")
    ax1.set_ylabel(f"Cena zamknięcia [{inst.currency}]")
    ax1.legend(loc="upper left", ncol=3, fontsize=8)

    ax2.plot(dates.iloc[sl], result.equity[sl], color=COLORS["equity"],
             lw=1.5, label="Wartość portfela $V_i$ (strategia MACD)")
    if bh_equity is not None:
        ax2.plot(dates.iloc[sl], bh_equity[sl], color=COLORS["bh"], lw=1.2,
                 ls="--", label="Odniesienie: „kup i trzymaj”")
    ax2.axhline(result.initial_capital, color=COLORS["neutral"], lw=0.8,
                ls=":", label=f"Kapitał początkowy $C_{{{start}}}$")
    ax2.set_title(f"E2: Wartość portfela $V_i$ – {inst.label}, "
                  f"R = {result.total_return_pct:+.2f}%")
    ax2.set_ylabel(f"Wartość portfela [{inst.currency}]")
    ax2.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f} tys."))
    ax2.legend(loc="upper left", ncol=3, fontsize=8.5)
    _date_axis(ax2)
    ax2.set_xlim(dates.iloc[start], dates.iloc[-1])
    return save_figure(fig, path)


# ----------------------------------------------------------------------------
# Część eksperymentalna
# ----------------------------------------------------------------------------
def plot_experiment_global(table: pd.DataFrame, path: Path) -> Path:
    """G1: rozkład zmian ΔR [p.p.] oraz rozkład stóp zwrotu trzech strategii."""
    diff = table["delta_pp"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.9),
                                   gridspec_kw={"width_ratios": [1.15, 1]})
    changed = diff[diff.abs() > 1e-9]
    clip = 60.0
    bins = np.arange(-clip, clip + 5, 5)
    ax1.hist(changed.clip(-clip, clip)[changed > 0], bins=bins,
             color=COLORS["buy"], alpha=0.8, label=f"Poprawa ({(diff > 1e-9).sum()})")
    ax1.hist(changed.clip(-clip, clip)[changed < 0], bins=bins,
             color=COLORS["sell"], alpha=0.8,
             label=f"Pogorszenie ({(diff < -1e-9).sum()})")
    n_same = int((diff.abs() <= 1e-9).sum())
    ax1.axvline(diff.mean(), color="black", ls="--", lw=1.2,
                label=f"Średnia ΔR = {diff.mean():+.2f} p.p.")
    ax1.axvline(diff.median(), color=COLORS["rsi"], ls="-.", lw=1.2,
                label=f"Mediana ΔR = {diff.median():+.2f} p.p.")
    ax1.plot([], [], " ", label=f"Bez zmian: {n_same} spółek")
    ax1.set_title("G1a: Zmiana stopy zwrotu ΔR = R_mod − R_MACD")
    ax1.set_xlabel(f"ΔR [p.p.] (wartości obcięte do ±{clip:.0f})")
    ax1.set_ylabel("Liczba spółek")
    ax1.legend(loc="upper left", fontsize=8.5)

    data = [table["r_base"], table["r_mod"], table["r_bh"]]
    labels = ["MACD\n(12, 26, 9)", "MACD +\nfiltr RSI", "Kup\ni trzymaj"]
    box = ax2.boxplot(data, tick_labels=labels, showfliers=False,
                      whis=(5, 95), patch_artist=True, widths=0.55)
    for patch, color in zip(box["boxes"],
                            [COLORS["macd"], COLORS["rsi"], COLORS["bh"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
    for med in box["medians"]:
        med.set_color("black")
    means = [d.mean() for d in data]
    ax2.scatter(range(1, 4), means, marker="D", color="black", zorder=5,
                label="Średnia")
    for k, mean in enumerate(means, start=1):
        ax2.annotate(f"{mean:.1f}%", (k, mean), xytext=(14, 0),
                     textcoords="offset points", va="center", fontsize=9)
    ax2.axhline(0, color=COLORS["neutral"], lw=0.8, ls=":")
    ax2.set_title(f"G1b: Stopy zwrotu R ({len(table)} spółek)")
    ax2.set_ylabel("Stopa zwrotu R [%]")
    ax2.legend(handles=[
        Line2D([], [], marker="D", color="black", ls="", label="Średnia"),
        Line2D([], [], color="black", label="Mediana"),
        Patch(facecolor="0.8", label="Kwartyle, wąsy: P5–P95"),
    ], loc="upper left", fontsize=8.5)
    fig.tight_layout()
    return save_figure(fig, path)


def plot_experiment_blocked(
    blocked: pd.DataFrame,
    kept_returns: pd.Series,
    sensitivity: pd.DataFrame,
    path: Path,
) -> Path:
    """G2: transakcje zablokowane vs pozostawione oraz wrażliwość na próg RSI."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.7))
    closed = blocked["return_pct"].dropna()
    bins = np.arange(-30, 32, 3)
    ax1.hist(kept_returns.clip(-30, 30), bins=bins, density=True,
             color=COLORS["macd"], alpha=0.45,
             label=f"Wykonane przez oba ({len(kept_returns)}), "
                   f"śr. r = {kept_returns.mean():+.2f}%")
    ax1.hist(closed.clip(-30, 30), bins=bins, density=True, histtype="step",
             lw=2, color=COLORS["sell"],
             label=f"Zablokowane filtrem ({len(closed)}), "
                   f"śr. r = {closed.mean():+.2f}%")
    ax1.axvline(0, color="black", lw=0.8)
    ax1.set_title("G2a: Stopy zwrotu r transakcji MACD")
    ax1.set_xlabel("Stopa zwrotu zamkniętej transakcji r [%] (obcięte do ±30)")
    ax1.set_ylabel("Gęstość [1/%]")
    ax1.legend(loc="upper left", fontsize=8)

    thr = sensitivity["prog"].to_numpy()
    ax2.plot(thr, sensitivity["mean_delta_pp"], marker="o",
             color="black", label="Średnia ΔR")
    ax2.plot(thr, sensitivity["median_delta_pp"], marker="s",
             color=COLORS["rsi"], ls="--", label="Mediana ΔR")
    ax2.axhline(0, color=COLORS["neutral"], lw=0.8, ls=":")
    ax2.set_title("G2b: Wrażliwość na próg filtru RSI")
    ax2.set_xlabel("Próg wykupienia RSI$_{max}$ (kupno tylko gdy RSI < próg)")
    ax2.set_ylabel("ΔR względem MACD [p.p.]")
    ax2.invert_xaxis()
    ax2.legend(loc="lower left", fontsize=8.5)
    fig.tight_layout()
    return save_figure(fig, path)


def plot_local_example(
    df: pd.DataFrame,
    base: SimulationResult,
    mod: SimulationResult,
    rsi: pd.Series,
    rsi_max: float,
    blocked_row: pd.Series,
    inst: Instrument,
    tag: str,
    delta_pp: float,
    path: Path,
    margin: int = 30,
) -> Path:
    """H: lokalny przykład decyzji zablokowanej przez filtr RSI."""
    sig_i = int(blocked_row["signal_index"])
    end_i = int(blocked_row["sell_index"])
    lo, hi = max(sig_i - margin, 0), min(end_i + margin, len(df) - 1)
    window = slice(lo, hi + 1)
    dates, close = df["date"], df["close"]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 4.9), sharex=True,
        gridspec_kw={"height_ratios": [1.6, 1], "hspace": 0.12},
    )
    ax1.plot(dates.iloc[window], close.iloc[window], color=COLORS["price"],
             lw=1.4, label="Cena zamknięcia $x_i$")
    ax1.axvspan(dates.iloc[int(blocked_row["buy_index"])], dates.iloc[end_i],
                color=COLORS["blocked"], zorder=0,
                label="Transakcja MACD pominięta przez filtr")

    def in_window(t):
        return lo <= t.exec_index <= hi

    # Duże puste znaczniki – algorytm podstawowy, małe pełne – zmodyfikowany.
    for t in filter(in_window, base.trades):
        is_buy = t.side == BUY
        ax1.scatter(dates.iloc[t.exec_index], t.price,
                    marker="^" if is_buy else "v", s=70,
                    facecolor="white",
                    edgecolor=COLORS["buy"] if is_buy else COLORS["sell"],
                    lw=1.6, zorder=5)
    for t in filter(in_window, mod.trades):
        is_buy = t.side == BUY
        ax1.scatter(dates.iloc[t.exec_index], t.price,
                    marker="^" if is_buy else "v", s=28,
                    color=COLORS["buy"] if is_buy else COLORS["sell"],
                    zorder=6)

    r = blocked_row["return_pct"]
    ax1.text(0.99, 0.05,
             f"Pominięta transakcja MACD: r = {r:+.2f}%\n"
             f"RSI w chwili sygnału kupna = {blocked_row['rsi_at_signal']:.1f}"
             f" ≥ {rsi_max:.0f}\nZmiana wyniku spółki: ΔR = {delta_pp:+.2f} p.p.",
             transform=ax1.transAxes, ha="right", va="bottom", fontsize=9.5,
             bbox={"boxstyle": "round", "fc": "white", "ec": "0.5"}, zorder=8)
    ax1.set_title(f"{tag}: {inst.label} – decyzja zmieniona przez filtr RSI "
                  f"({_fmt_date(dates.iloc[lo])} – {_fmt_date(dates.iloc[hi])})")
    ax1.set_ylabel(f"Cena [{inst.currency}]")
    legend = ax1.get_legend_handles_labels()[0] + [
        Line2D([], [], marker="^", ls="", mfc="white", mec=COLORS["buy"],
               mew=1.6, ms=8, label="Kupno – MACD"),
        Line2D([], [], marker="v", ls="", mfc="white", mec=COLORS["sell"],
               mew=1.6, ms=8, label="Sprzedaż – MACD"),
        Line2D([], [], marker="^", ls="", color=COLORS["buy"], ms=5,
               label="Kupno – MACD + RSI"),
        Line2D([], [], marker="v", ls="", color=COLORS["sell"], ms=5,
               label="Sprzedaż – MACD + RSI"),
    ]
    ax1.legend(handles=legend, loc="upper left", ncol=3, fontsize=8)
    ymin, ymax = ax1.get_ylim()
    ax1.set_ylim(ymin, ymax + 0.3 * (ymax - ymin))

    ax2.plot(dates.iloc[window], rsi.iloc[window], color=COLORS["rsi"],
             lw=1.3, label="RSI(14)")
    ax2.axhline(rsi_max, color=COLORS["sell"], ls="--", lw=1,
                label=f"Próg wykupienia {rsi_max:.0f}")
    ax2.scatter(dates.iloc[sig_i], rsi.iloc[sig_i], marker="X", s=90,
                color="black", zorder=5, label="Zablokowany sygnał kupna")
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("RSI [pkt]")
    ax2.legend(loc="lower left", ncol=3, fontsize=8)
    _date_axis(ax2, "%Y-%m-%d")
    ax2.set_xlim(dates.iloc[lo], dates.iloc[hi])
    return save_figure(fig, path)
