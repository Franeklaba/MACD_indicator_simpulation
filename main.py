"""Punkt wejścia programu: symulacja wskaźnika MACD (Projekt 1).

Przykłady:
    python main.py podstawowa            # wykresy A1, B1, C1, C2, D1, E1, E2
    python main.py eksperyment           # wykresy G1, G2, H1, H2 + tabele
    python main.py wszystko --out wyniki # obie części
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from src.pipeline import (
    RSI_MAX,
    BasicConfig,
    ExperimentConfig,
    run_basic,
    run_experimental,
)

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_BASIC_FILE = PROJECT_DIR / "dane_podstawowe" / "AAPL.csv"
DEFAULT_DATA_DIR = Path(os.environ.get("MACD_DATA_DIR", PROJECT_DIR / "data"))
DEFAULT_OUT_DIR = PROJECT_DIR / "wyniki"


def build_parser() -> argparse.ArgumentParser:
    """Tworzy parser argumentów wiersza poleceń."""
    parser = argparse.ArgumentParser(
        description="Symulacja strategii MACD(12, 26, 9) i jej modyfikacji.",
    )
    parser.add_argument(
        "tryb", choices=["podstawowa", "eksperyment", "wszystko"],
        help="która część projektu ma zostać uruchomiona",
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT_DIR,
        help="katalog wyników (wykresy PDF, CSV, TeX); domyślnie: %(default)s",
    )
    parser.add_argument(
        "--delay", "-d", type=int, choices=[0, 1], default=1,
        help="opóźnienie wykonania transakcji d (domyślnie 1)",
    )
    parser.add_argument(
        "--plik", type=Path, default=DEFAULT_BASIC_FILE,
        help="plik CSV części podstawowej; domyślnie: %(default)s",
    )
    parser.add_argument(
        "--d1", default="max_gain",
        help="wybór transakcji D1: max_gain, max_loss lub numer transakcji",
    )
    parser.add_argument(
        "--dane", type=Path, default=DEFAULT_DATA_DIR,
        help="katalog z plikami CSV z dane_pro1.zip (lub zmienna "
             "MACD_DATA_DIR); domyślnie: %(default)s",
    )
    parser.add_argument(
        "--rsi-max", type=float, default=RSI_MAX,
        help="próg wykupienia filtru RSI (domyślnie %(default)s)",
    )
    parser.add_argument(
        "--lokalny", default=None,
        help="ticker, z którego wybierane są przykłady lokalne H1/H2",
    )
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="szczegółowe komunikaty diagnostyczne")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Uruchamia wybraną część projektu; zwraca kod wyjścia procesu."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    try:
        if args.tryb in ("podstawowa", "wszystko"):
            run_basic(BasicConfig(args.plik, args.out, args.delay, args.d1))
        if args.tryb in ("eksperyment", "wszystko"):
            run_experimental(ExperimentConfig(
                args.dane, args.out, args.delay, args.rsi_max, args.lokalny,
            ))
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logging.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
