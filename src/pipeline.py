"""Orkiestracja obliczeń: część podstawowa i eksperymentalna projektu.

Moduł łączy warstwy: dane -> wskaźniki/strategie -> symulacja -> metryki
-> eksport (CSV/TXT/TeX) i wykresy (PDF).
"""

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src import visualizer as vis
from src.backtester import CAPITAL_UNITS, START_INDEX, simulate
from src.data_loader import DataLoader
from src.experiment import ExperimentResult, run_experiment
from src.metrics import SummaryStats, format_summary, summarize
from src.strategies import FilteredMacdStrategy, MacdStrategy

log = logging.getLogger(__name__)

INSTRUMENT_NAMES = {
    "AAPL": "Apple Inc.",
    "ABNB": "Airbnb, Inc.",
    "MSFT": "Microsoft Corp.",
    "NFLX": "Netflix, Inc.",
    "NVDA": "NVIDIA Corp.",
    "PLTR": "Palantir Technologies Inc.",
    "QCOM": "Qualcomm Inc.",
    "SMCI": "Super Micro Computer, Inc.",
}
RSI_MAX = 70.0
SENSITIVITY_THRESHOLDS = (80.0, 75.0, 70.0, 65.0, 60.0, 55.0, 50.0)


@dataclass(frozen=True)
class BasicConfig:
    """Konfiguracja części podstawowej."""

    data_file: Path
    out_dir: Path
    delay: int = 1
    d1_criterion: str = "max_gain"


@dataclass(frozen=True)
class ExperimentConfig:
    """Konfiguracja części eksperymentalnej."""

    data_dir: Path
    out_dir: Path
    delay: int = 1
    rsi_max: float = RSI_MAX
    local_ticker: str | None = None


# ----------------------------------------------------------------------------
# Narzędzia eksportu
# ----------------------------------------------------------------------------
def _pl(value: float, digits: int = 2) -> str:
    """Formatuje liczbę z przecinkiem dziesiętnym (konwencja polska)."""
    return f"{value:.{digits}f}".replace(".", ",")


def _pl_money(value: float) -> str:
    """Formatuje kwotę z odstępem tysięcy (LaTeX: \\,) i przecinkiem."""
    text = f"{abs(value):,.2f}".replace(",", "\\,").replace(".", ",")
    return ("-" if value < 0 else "") + text


class TexMacros:
    """Zbiera wartości liczbowe i zapisuje je jako makra LaTeX."""

    def __init__(self) -> None:
        self._items: dict[str, str] = {}

    def add(self, name: str, value: str) -> None:
        """Dodaje makro ``\\name`` (nazwa wyłącznie z liter)."""
        if not name.isalpha():
            raise ValueError(f"Nazwa makra LaTeX musi składać się z liter: {name}")
        self._items[name] = value

    def write(self, path: Path) -> Path:
        """Zapisuje plik z definicjami makr (\\providecommand)."""
        lines = ["% Plik wygenerowany automatycznie przez main.py – nie edytować."]
        lines += [
            f"\\providecommand{{\\{k}}}{{{v}}}" for k, v in self._items.items()
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path


def _pct_or_na(value: float | None) -> str:
    return "nie dotyczy" if value is None else f"{_pl(value)}\\%"


def _sync_presentation_figures(out_dir: Path) -> None:
    """Kopiuje wykresy PDF i makra TeX do katalogu presentation/figures."""
    pres_fig = Path(__file__).resolve().parents[1] / "presentation" / "figures"
    if pres_fig.resolve() == out_dir.resolve():
        return
    pres_fig.mkdir(parents=True, exist_ok=True)
    for pdf in (out_dir / "wykresy").glob("*.pdf"):
        shutil.copy2(pdf, pres_fig / pdf.name)
    for tex in out_dir.glob("*.tex"):
        shutil.copy2(tex, pres_fig / tex.name)


# ----------------------------------------------------------------------------
# Część podstawowa
# ----------------------------------------------------------------------------
def _select_d1_trade(result, criterion: str):
    """Wybiera zamkniętą transakcję do wykresu D1 według kryterium."""
    closed = result.closed_trades
    if not closed:
        raise RuntimeError(
            "Brak zamkniętych transakcji – zmień dane wejściowe części "
            "podstawowej (wymóg specyfikacji)."
        )
    if criterion == "max_gain":
        return max(closed, key=lambda t: t.return_pct)
    if criterion == "max_loss":
        return min(closed, key=lambda t: t.return_pct)
    if criterion.isdigit():
        idx = int(criterion)
        if not 0 <= idx < len(closed):
            raise ValueError(f"Numer transakcji D1 poza zakresem 0..{len(closed) - 1}")
        return closed[idx]
    raise ValueError(f"Nieznane kryterium wyboru transakcji D1: {criterion}")


def run_basic(cfg: BasicConfig) -> SummaryStats:
    """Część podstawowa: MACD(12, 26, 9), symulacja, statystyki i wykresy A1–E2."""
    loader = DataLoader(cfg.data_file.parent)
    stock = loader.load_stock(cfg.data_file)
    df, close = stock.df, stock.df["close"]
    inst = vis.Instrument(stock.ticker, INSTRUMENT_NAMES.get(stock.ticker, ""))
    log.info("Część podstawowa: %s, L = %d, d = %d", inst.label, len(df), cfg.delay)

    output = MacdStrategy().generate(close)
    result = simulate(close, output.signals, delay=cfg.delay)
    stats = summarize(result)
    trade_d1 = _select_d1_trade(result, cfg.d1_criterion)

    out = cfg.out_dir
    fig_dir = out / "wykresy"
    out.mkdir(parents=True, exist_ok=True)

    # --- Eksport danych ---
    pd.DataFrame({
        "date": df["date"], "close": close,
        "macd": output.macd.macd, "signal": output.macd.signal,
        "histogram": output.macd.histogram, "sygnal": output.signals,
        "cash": result.cash, "position_qty": result.position_qty,
        "equity": result.equity,
    }).to_csv(out / "podstawowa_przebieg.csv", index_label="i")
    trades = pd.DataFrame([{
        "kupno_sygnal_i": t.buy.signal_index, "kupno_i": t.buy.exec_index,
        "kupno_data": df["date"].iloc[t.buy.exec_index].date(),
        "kupno_cena": t.buy.price,
        "sprzedaz_sygnal_i": t.sell.signal_index, "sprzedaz_i": t.sell.exec_index,
        "sprzedaz_data": df["date"].iloc[t.sell.exec_index].date(),
        "sprzedaz_cena": t.sell.price, "r_proc": t.return_pct,
    } for t in result.closed_trades])
    trades.to_csv(out / "podstawowa_transakcje.csv", index=False)
    report = format_summary(stats, inst.currency)
    (out / "podstawowa_podsumowanie.txt").write_text(
        f"{inst.label}, d = {cfg.delay}\n{report}\n", encoding="utf-8"
    )
    (out / "podstawowa_podsumowanie.json").write_text(
        json.dumps(stats.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n=== Część podstawowa: {inst.label} (d = {cfg.delay}) ===\n{report}")

    # --- Wykresy A1–E2 ---
    bh_equity = np.full(len(close), np.nan)
    bh_equity[START_INDEX:] = CAPITAL_UNITS * close.to_numpy()[START_INDEX:]
    vis.plot_a1_price(df, inst, fig_dir / "A1_cena.pdf")
    vis.plot_b1_price_macd(df, output.macd, inst, fig_dir / "B1_cena_macd.pdf")
    vis.plot_c_signals_histogram(df, output.macd, output.signals, inst,
                                 fig_dir / "C1_C2_sygnaly_histogram.pdf")
    vis.plot_d1_trade(df, output.macd, trade_d1, inst,
                      fig_dir / "D1_transakcja.pdf")
    vis.plot_e_simulation(df, result, inst, fig_dir / "E1_E2_symulacja.pdf",
                          bh_equity=bh_equity)
    log.info("Zapisano wykresy A1–E2 w %s", fig_dir)

    # --- Makra LaTeX ---
    tex = TexMacros()
    dates = df["date"]
    r_bh = 100.0 * (close.iloc[-1] / close.iloc[START_INDEX] - 1.0)
    tex.add("Ticker", inst.ticker)
    tex.add("InstName", inst.name or inst.ticker)
    tex.add("Lobs", str(len(df)))
    tex.add("Lmax", str(len(df) - 1))
    tex.add("DateFirst", str(dates.iloc[0].date()))
    tex.add("DateLast", str(dates.iloc[-1].date()))
    tex.add("DateStart", str(dates.iloc[START_INDEX].date()))
    tex.add("PriceFirst", _pl(close.iloc[0]))
    tex.add("PriceLast", _pl(close.iloc[-1]))
    tex.add("PriceMin", _pl(close.min()))
    tex.add("PriceMax", _pl(close.max()))
    tex.add("PriceStart", _pl(close.iloc[START_INDEX]))
    tex.add("Delay", str(cfg.delay))
    tex.add("CapStart", _pl_money(stats.initial_capital))
    tex.add("CapEnd", _pl_money(stats.final_equity))
    tex.add("Profit", ("+" if stats.profit >= 0 else "") + _pl_money(stats.profit))
    tex.add("Rtot", f"{'+' if stats.total_return_pct >= 0 else ''}"
                    f"{_pl(stats.total_return_pct)}")
    tex.add("RBH", f"{'+' if r_bh >= 0 else ''}{_pl(r_bh)}")
    tex.add("Nclosed", str(stats.n_closed))
    tex.add("Nwin", str(stats.n_profitable))
    tex.add("Nloss", str(stats.n_losing))
    tex.add("WinRate", _pct_or_na(stats.win_rate_pct))
    tex.add("MaxLoss", _pct_or_na(stats.max_loss_pct))
    tex.add("MaxGain", _pct_or_na(stats.max_gain_pct))
    tex.add("MeanTrade", _pct_or_na(stats.mean_trade_return_pct))
    tex.add("Nsignals", str(stats.n_signals))
    tex.add("Nexecuted", str(stats.n_executed))
    tex.add("NbuySig", str(int((output.signals == 1).sum())))
    tex.add("NsellSig", str(int((output.signals == -1).sum())))
    if stats.open_position:
        buy_date = dates.iloc[stats.open_position_buy_index].date()
        buy_price = _pl(stats.open_position_buy_price)
        tex.add("OpenPos", f"tak -- kupno {buy_date} po {buy_price} USD")
    else:
        tex.add("OpenPos", "brak (cała wartość w gotówce)")

    for prefix, t in (("DBuy", trade_d1.buy), ("DSell", trade_d1.sell)):
        tex.add(f"{prefix}SigDate", str(dates.iloc[t.signal_index].date()))
        tex.add(f"{prefix}Date", str(dates.iloc[t.exec_index].date()))
        tex.add(f"{prefix}Price", _pl(t.price))
        tex.add(f"{prefix}Idx", str(t.exec_index))
        tex.add(f"{prefix}SigIdx", str(t.signal_index))
    tex.add("DReturn", f"{'+' if trade_d1.return_pct >= 0 else ''}"
                       f"{_pl(trade_d1.return_pct)}")
    tex.add("DLength", str(trade_d1.sell.exec_index - trade_d1.buy.exec_index))
    tex.write(out / "liczby_podstawowa.tex")
    _sync_presentation_figures(out)
    return stats


# ----------------------------------------------------------------------------
# Część eksperymentalna
# ----------------------------------------------------------------------------
def _pick_local_examples(res: ExperimentResult, ticker: str | None):
    """Wybiera przykłady lokalne: największa uniknięta strata i utracony zysk.

    Dla przykładu pozytywnego (H1) preferowane są spółki, dla których
    modyfikacja poprawiła również łączną stopę zwrotu portfela (ΔR > 0),
    a dla przykładu negatywnego (H2) – spółki z pogorszeniem (ΔR < 0).
    """
    closed = res.blocked.dropna(subset=["return_pct"]).copy()
    closed["delta_pp"] = closed["ticker"].map(res.table["delta_pp"])
    if ticker is not None:
        subset = closed[closed["ticker"] == ticker]
        if not subset.empty:
            closed = subset
    if closed.empty:
        raise RuntimeError("Filtr nie zablokował żadnej zamkniętej transakcji.")
    pos_pool = closed[(closed["delta_pp"] > 0) & (closed["sell_index"] <= 974)]
    neg_pool = closed[(closed["delta_pp"] < 0) & (closed["sell_index"] <= 974)]
    better_pool = pos_pool if not pos_pool.empty else closed
    worse_pool = neg_pool if not neg_pool.empty else closed
    better = better_pool.loc[better_pool["return_pct"].idxmin()]
    worse = worse_pool.loc[worse_pool["return_pct"].idxmax()]
    return better, worse


def run_experimental(cfg: ExperimentConfig) -> ExperimentResult:
    """Część eksperymentalna: MACD vs MACD + filtr RSI vs „kup i trzymaj”."""
    loader = DataLoader(cfg.data_dir)
    stocks = loader.load_all()
    if not stocks:
        raise FileNotFoundError(f"Brak plików CSV w katalogu {cfg.data_dir}")
    log.info("Część eksperymentalna: %d spółek, d = %d, RSI < %.0f",
             len(stocks), cfg.delay, cfg.rsi_max)

    base = MacdStrategy()
    mod = FilteredMacdStrategy(rsi_max=cfg.rsi_max, name="MACD + RSI")
    variants = {thr: FilteredMacdStrategy(rsi_max=thr)
                for thr in SENSITIVITY_THRESHOLDS}
    res = run_experiment(stocks, base, mod, cfg.delay, variants)
    res.sensitivity.insert(0, "prog", res.sensitivity.pop("wariant"))

    out = cfg.out_dir
    fig_dir = out / "wykresy"
    out.mkdir(parents=True, exist_ok=True)
    res.table.to_csv(out / "eksperyment_spolki.csv", float_format="%.4f")
    res.blocked.to_csv(out / "eksperyment_zablokowane.csv", index=False,
                       float_format="%.4f")
    res.sensitivity.to_csv(out / "eksperyment_wrazliwosc.csv", index=False,
                           float_format="%.4f")
    summary = res.global_summary()
    (out / "eksperyment_podsumowanie.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n=== Część eksperymentalna: wynik globalny ===")
    for key, value in summary.items():
        print(f"{key:18s} {value:10.2f}" if isinstance(value, float)
              else f"{key:18s} {value:10d}")
    print(res.sensitivity.round(2).to_string(index=False))

    # --- Wykresy G1, G2, H1, H2 ---
    vis.plot_experiment_global(res.table, fig_dir / "G1_wynik_globalny.pdf")
    vis.plot_experiment_blocked(res.blocked, res.kept_returns, res.sensitivity,
                                fig_dir / "G2_analiza_filtru.pdf")
    better, worse = _pick_local_examples(res, cfg.local_ticker)
    tex = TexMacros()
    for tag, row, prefix in (("H1", better, "HA"), ("H2", worse, "HB")):
        ticker = row["ticker"]
        prices = stocks[ticker].df["close"]
        base_out, mod_out = base.generate(prices), mod.generate(prices)
        base_res = simulate(prices, base_out.signals, cfg.delay)
        mod_res = simulate(prices, mod_out.signals, cfg.delay)
        delta = float(res.table.loc[ticker, "delta_pp"])
        inst = vis.Instrument(ticker, INSTRUMENT_NAMES.get(ticker, ""))
        vis.plot_local_example(
            stocks[ticker].df, base_res, mod_res, mod_out.extras["rsi"],
            cfg.rsi_max, row, inst, tag, delta,
            fig_dir / f"{tag}_przyklad_lokalny.pdf",
        )
        dates = stocks[ticker].df["date"]
        tex.add(f"{prefix}Ticker", ticker)
        tex.add(f"{prefix}SigDate", str(dates.iloc[int(row['signal_index'])].date()))
        tex.add(f"{prefix}BuyDate", str(dates.iloc[int(row['buy_index'])].date()))
        tex.add(f"{prefix}SellDate", str(dates.iloc[int(row['sell_index'])].date()))
        tex.add(f"{prefix}BuyPrice", _pl(row["buy_price"]))
        tex.add(f"{prefix}SellPrice", _pl(row["sell_price"]))
        tex.add(f"{prefix}Return", f"{row['return_pct']:+.2f}".replace(".", ","))
        tex.add(f"{prefix}Rsi", _pl(row["rsi_at_signal"], 1))
        tex.add(f"{prefix}Delta", f"{delta:+.2f}".replace(".", ","))
        tex.add(f"{prefix}Rbase", _pl(res.table.loc[ticker, "r_base"]))
        tex.add(f"{prefix}Rmod", _pl(res.table.loc[ticker, "r_mod"]))
    log.info("Zapisano wykresy G1, G2, H1, H2 w %s", fig_dir)

    # --- Makra LaTeX ---
    def signed(v: float) -> str:
        return f"{v:+.2f}".replace(".", ",")

    blocked_closed = res.blocked["return_pct"].dropna()
    t = res.table
    tex.add("XN", str(summary["n_tickers"]))
    tex.add("XRsiMax", f"{cfg.rsi_max:.0f}")
    tex.add("XMeanDelta", signed(summary["mean_delta_pp"]))
    tex.add("XMedianDelta", signed(summary["median_delta_pp"]))
    tex.add("XBetter", str(summary["n_better"]))
    tex.add("XWorse", str(summary["n_worse"]))
    tex.add("XSame", str(summary["n_same"]))
    for key, name in (("base", "Base"), ("mod", "Mod"), ("bh", "BH")):
        tex.add(f"XMean{name}", _pl(summary[f"mean_r_{key}"]))
        tex.add(f"XMedian{name}", _pl(summary[f"median_r_{key}"]))
    tex.add("XBaseBeatsBH", str(summary["base_beats_bh"]))
    tex.add("XModBeatsBH", str(summary["mod_beats_bh"]))
    tex.add("XTradesBase", _pl(t["trades_base"].mean(), 1))
    tex.add("XTradesMod", _pl(t["trades_mod"].mean(), 1))
    tex.add("XWinBase", _pl(t["win_base"].mean(), 1))
    tex.add("XWinMod", _pl(t["win_mod"].mean(), 1))
    tex.add("XMaxLossBase", _pl(t["max_loss_base"].mean(), 1))
    tex.add("XMaxLossMod", _pl(t["max_loss_mod"].mean(), 1))
    tex.add("XBlocked", str(len(res.blocked)))
    tex.add("XBlockedClosed", str(len(blocked_closed)))
    tex.add("XBlockedMean", signed(blocked_closed.mean()))
    tex.add("XBlockedWin", _pl(100.0 * (blocked_closed >= 0).mean(), 1))
    tex.add("XKeptMean", signed(res.kept_returns.mean()))
    tex.add("XKeptWin", _pl(100.0 * (res.kept_returns >= 0).mean(), 1))
    tex.add("XKept", str(len(res.kept_returns)))
    tex.write(out / "liczby_eksperyment.tex")

    sens_lines = [
        "% Wygenerowano automatycznie przez main.py",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}rrrrr@{}}",
        r"\toprule",
        r"$\text{RSI}_{\max}$ & Śr.~$\Delta R$ & Med.~$\Delta R$ "
        r"& $+ / - / =$ & Transakcje \\",
        r"\midrule",
    ]
    for _, row in res.sensitivity.iterrows():
        sens_lines.append(
            f"{row['prog']:.0f} & {signed(row['mean_delta_pp'])} & "
            f"{signed(row['median_delta_pp'])} & {int(row['n_better'])} / "
            f"{int(row['n_worse'])} / {int(row['n_same'])} & "
            f"{_pl(row['mean_trades'], 1)} \\\\"
        )
    sens_lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "tabela_wrazliwosc.tex").write_text(
        "\n".join(sens_lines) + "\n",
        encoding="utf-8",
    )
    _sync_presentation_figures(out)
    return res

