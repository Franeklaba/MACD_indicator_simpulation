"""Moduł do wczytywania i walidacji danych giełdowych z plików CSV."""

from dataclasses import dataclass
from pathlib import Path
import pandas as pd


REQUIRED_COLUMNS = ["date", "open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class StockData:
    """Struktura przechowująca notowania pojedynczej spółki."""
    ticker: str
    df: pd.DataFrame


class DataLoader:
    """Wczytuje i waliduje dane giełdowe z plików CSV."""

    def __init__(self, data_dir: str | Path = "data") -> None:
        self.data_dir = Path(data_dir)

    def load_stock(self, source: str | Path) -> StockData:
        """Wczytuje dane pojedynczej spółki z podanej ścieżki lub tickera."""
        path = Path(source)
        if not path.is_file():
            path = self.data_dir / f"{source}.csv"
            if not path.is_file():
                raise FileNotFoundError(f"Nie znaleziono pliku dla: {source}")

        df = pd.read_csv(path)
        df.columns = [col.strip().lower() for col in df.columns]

        missing = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(
                f"Plik {path.name} nie zawiera wymaganych kolumn: {missing}"
            )

        df["date"] = pd.to_datetime(df["date"])
        df = df[REQUIRED_COLUMNS].sort_values("date").reset_index(drop=True)

        self._validate(df, path.name)
        return StockData(ticker=path.stem.upper(), df=df)

    def load_all(self) -> dict[str, StockData]:
        """Wczytuje wszystkie pliki CSV z katalogu danych do słownika."""
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"Katalog nie istnieje: {self.data_dir}")

        stocks: dict[str, StockData] = {}
        for file_path in sorted(self.data_dir.glob("*.csv")):
            stock = self.load_stock(file_path)
            stocks[stock.ticker] = stock
        return stocks

    @staticmethod
    def _validate(df: pd.DataFrame, file_name: str) -> None:
        """Weryfikuje zgodność danych ze specyfikacją projektu numerycznego."""
        if not (1000 <= len(df) <= 1200):
            raise ValueError(
                f"Plik {file_name}: nieprawidłowa liczba obserwacji "
                f"L={len(df)} (wymagane 1000-1200)."
            )
        if df.isna().any().any():
            raise ValueError(f"Plik {file_name}: wykryto brakujące dane (NaN).")
        if df["date"].duplicated().any():
            raise ValueError(f"Plik {file_name}: wykryto zduplikowane daty sesji.")
        if (df[["open", "high", "low", "close"]] <= 0).any().any():
            raise ValueError(f"Plik {file_name}: wykryto niedodatnie ceny.")
