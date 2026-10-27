"""Module for loading and validating stock market data for MACD indicator simulations.

Responsible for reading quotes of individual companies and batch processing
a dataset of companies (e.g. Nasdaq-100) from CSV files, numerical validation,
and providing them as consistent data structures.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Union
import numpy as np
import pandas as pd


# Required columns in stock market CSV files
REQUIRED_COLUMNS: tuple[str, ...] = ("date", "open", "high", "low", "close", "volume")

# Default allowable range for observation count L
MIN_OBSERVATIONS: int = 1000
MAX_OBSERVATIONS: int = 1200


@dataclass
class StockData:
    """Structure storing time-series quotations for a single stock.

    Attributes:
        ticker (str): Stock ticker symbol (e.g., 'AAPL').
        data (pd.DataFrame): Quotation table with columns date, open, high, low,
            close, volume, indexed from 0 to L-1 sorted chronologically.
    """

    ticker: str
    data: pd.DataFrame

    def __post_init__(self) -> None:
        """Ensure that the DataFrame index is a RangeIndex from 0 to L-1."""
        if not isinstance(self.data.index, pd.RangeIndex):
            self.data = self.data.reset_index(drop=True)

    @property
    def close(self) -> pd.Series:
        """Closing prices x_i as pd.Series with index 0 .. L-1."""
        return self.data["close"]

    @property
    def open(self) -> pd.Series:
        """Opening prices as pd.Series with index 0 .. L-1."""
        return self.data["open"]

    @property
    def high(self) -> pd.Series:
        """High prices as pd.Series with index 0 .. L-1."""
        return self.data["high"]

    @property
    def low(self) -> pd.Series:
        """Low prices as pd.Series with index 0 .. L-1."""
        return self.data["low"]

    @property
    def volume(self) -> pd.Series:
        """Trading volume as pd.Series with index 0 .. L-1."""
        return self.data["volume"]

    @property
    def dates(self) -> pd.Series:
        """Trading session dates as pd.Series."""
        return self.data["date"]

    @property
    def prices(self) -> np.ndarray:
        """Vector of closing prices x_i as a 1D numpy.ndarray."""
        return self.data["close"].to_numpy(dtype=np.float64)

    @property
    def length(self) -> int:
        """Number of observations L."""
        return len(self.data)

    def __len__(self) -> int:
        return self.length

    def __repr__(self) -> str:
        start_date = str(self.data["date"].iloc[0]) if len(self.data) > 0 else "N/A"
        end_date = str(self.data["date"].iloc[-1]) if len(self.data) > 0 else "N/A"
        return (
            f"StockData(ticker='{self.ticker}', observations={self.length}, "
            f"start='{start_date}', end='{end_date}')"
        )


class StockDataset:
    """Collection of stock market data for multiple tickers for batch analysis.

    Provides dictionary-like access by ticker, iteration, and generation
    of aggregated price matrices (e.g., for experimental part P3).
    """

    def __init__(self, stocks: Optional[Dict[str, StockData]] = None) -> None:
        self._stocks: Dict[str, StockData] = stocks if stocks is not None else {}

    def add(self, stock: StockData) -> None:
        """Add stock quotations to the dataset."""
        self._stocks[stock.ticker] = stock

    def get(self, ticker: str) -> Optional[StockData]:
        """Return data for the given ticker or None if not found."""
        return self._stocks.get(ticker)

    @property
    def tickers(self) -> List[str]:
        """List of all available tickers sorted alphabetically."""
        return sorted(self._stocks.keys())

    def get_close_matrix(self) -> pd.DataFrame:
        """Create a DataFrame of closing prices, where columns are stock tickers.

        The index is the common sample index 0 .. L-1.
        """
        matrix_dict = {ticker: stock.close for ticker, stock in self._stocks.items()}
        return pd.DataFrame(matrix_dict)

    def summary(self) -> pd.DataFrame:
        """Return a summary table of all loaded stocks."""
        records = []
        for ticker in self.tickers:
            stock = self._stocks[ticker]
            records.append({
                "ticker": ticker,
                "observations": stock.length,
                "start_date": str(stock.dates.iloc[0]),
                "end_date": str(stock.dates.iloc[-1]),
                "initial_close": float(stock.close.iloc[0]),
                "final_close": float(stock.close.iloc[-1]),
            })
        return pd.DataFrame(records)

    def __getitem__(self, ticker: str) -> StockData:
        if ticker not in self._stocks:
            raise KeyError(f"Ticker '{ticker}' not found in dataset.")
        return self._stocks[ticker]

    def __contains__(self, ticker: str) -> bool:
        return ticker in self._stocks

    def __len__(self) -> int:
        return len(self._stocks)

    def __iter__(self) -> Iterator[StockData]:
        return iter(self._stocks.values())

    def items(self):
        return self._stocks.items()

    def keys(self):
        return self._stocks.keys()

    def values(self):
        return self._stocks.values()

    def __repr__(self) -> str:
        return f"StockDataset(stocks_count={len(self._stocks)})"


class DataLoader:
    """Class responsible for discovering, loading, and validating CSV files."""

    def __init__(self, data_dir: Optional[Union[str, Path]] = None) -> None:
        """Initialize DataLoader with a specified or default data directory.

        Parameters:
            data_dir: Path to directory containing CSV files. If omitted,
                standard project locations are searched:
                'data', 'dane_pro1', and '../data'.
        """
        self.data_dir = self._resolve_data_dir(data_dir)

    @staticmethod
    def _resolve_data_dir(data_dir: Optional[Union[str, Path]]) -> Path:
        """Resolve data directory based on provided path or default candidate locations."""
        if data_dir is not None:
            path = Path(data_dir).expanduser().resolve()
            if not path.is_dir():
                raise FileNotFoundError(f"Data directory does not exist: {path}")
            return path

        # Candidate locations for data directory
        current_file_dir = Path(__file__).resolve().parent
        candidates = [
            current_file_dir.parent / "data",
            current_file_dir.parent / "dane_pro1",
            current_file_dir.parent.parent / "data",
            current_file_dir.parent.parent / "dane_pro1",
            Path.cwd() / "data",
            Path.cwd() / "dane_pro1",
        ]

        for cand in candidates:
            if cand.is_dir() and any(cand.glob("*.csv")):
                return cand.resolve()

        # If none match, return 'data' directory in project root
        default_dir = (current_file_dir.parent / "data").resolve()
        return default_dir

    def _resolve_file_path(self, source: Union[str, Path]) -> tuple[Path, str]:
        """Resolve CSV file path and stock ticker symbol."""
        source_path = Path(source)
        if source_path.is_file():
            ticker = source_path.stem.upper()
            return source_path.resolve(), ticker

        # Treat 'source' as ticker symbol or filename in data_dir
        filename = source if source.lower().endswith(".csv") else f"{source}.csv"
        target_path = self.data_dir / filename
        if not target_path.is_file():
            # Try case-insensitive matching
            matched = list(self.data_dir.glob(f"{source_path.stem}.[cC][sS][vV]"))
            if matched:
                target_path = matched[0]
            else:
                raise FileNotFoundError(
                    f"File not found for symbol '{source}' in directory {self.data_dir}"
                )

        ticker = target_path.stem.upper()
        return target_path.resolve(), ticker

    def load_stock(
        self,
        source: Union[str, Path],
        validate: bool = True,
        min_obs: int = MIN_OBSERVATIONS,
        max_obs: int = MAX_OBSERVATIONS,
    ) -> StockData:
        """Load quotations for a single stock from a CSV file and perform validation.

        Parameters:
            source: Path to CSV file or stock ticker symbol (e.g. 'AAPL' or 'AAPL.csv').
            validate: Whether to perform full numerical validation.
            min_obs: Minimum required number of observations L (default 1000).
            max_obs: Maximum allowable number of observations L (default 1200).

        Returns:
            StockData object containing the stock data.
        """
        file_path, ticker = self._resolve_file_path(source)

        try:
            df = pd.read_csv(file_path)
        except Exception as exc:
            raise IOError(f"Error reading CSV file {file_path}: {exc}") from exc

        # Standardize column names
        df.columns = [str(col).strip().lower() for col in df.columns]

        # Verify presence of required columns
        missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"File {file_path.name} is missing required columns: {missing_cols}"
            )

        # Select and reorder required columns
        df = df[list(REQUIRED_COLUMNS)].copy()

        # Convert numeric types
        for num_col in ("open", "high", "low", "close", "volume"):
            df[num_col] = pd.to_numeric(df[num_col], errors="coerce")

        # Sort chronologically from oldest to newest dates
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values(by="date", ascending=True).reset_index(drop=True)

        if validate:
            self._validate_dataframe(df, file_name=file_path.name, min_obs=min_obs, max_obs=max_obs)

        return StockData(ticker=ticker, data=df)

    @staticmethod
    def _validate_dataframe(
        df: pd.DataFrame,
        file_name: str,
        min_obs: int,
        max_obs: int,
    ) -> None:
        """Validate DataFrame against strict numerical project requirements.

        Requirements:
        1. Number of observations L in range [min_obs, max_obs] (standard [1000, 1200]).
        2. No missing values (NaN/None).
        3. No duplicate trading dates.
        4. Strict chronological ordering (oldest to newest).
        5. All prices (open, high, low, close) strictly positive (> 0).
        """
        obs_count = len(df)
        if not (min_obs <= obs_count <= max_obs):
            raise ValueError(
                f"File '{file_name}': invalid number of observations L={obs_count}. "
                f"Required range is [{min_obs}, {max_obs}]."
            )

        # Missing values
        if df.isna().any().any():
            nan_summary = df.isna().sum().to_dict()
            raise ValueError(
                f"File '{file_name}': detected missing values in data: {nan_summary}"
            )

        # Duplicate dates
        if df["date"].duplicated().any():
            dups = df["date"][df["date"].duplicated()].tolist()
            raise ValueError(
                f"File '{file_name}': detected duplicate trading dates: {dups}"
            )

        # Chronological order
        if not df["date"].is_monotonic_increasing:
            raise ValueError(
                f"File '{file_name}': time series is not strictly chronologically ordered."
            )

        # Positive prices
        for col in ("open", "high", "low", "close"):
            if (df[col] <= 0).any():
                non_pos_count = (df[col] <= 0).sum()
                raise ValueError(
                    f"File '{file_name}': column '{col}' contains {non_pos_count} "
                    f"non-positive values (strictly positive prices > 0 required)."
                )

    def load_stocks(
        self,
        tickers: Sequence[str],
        validate: bool = True,
    ) -> StockDataset:
        """Load a list of specified tickers into a StockDataset structure."""
        dataset = StockDataset()
        for ticker in tickers:
            stock = self.load_stock(ticker, validate=validate)
            dataset.add(stock)
        return dataset

    def load_all(
        self,
        pattern: str = "*.csv",
        validate: bool = True,
        silent: bool = False,
    ) -> StockDataset:
        """Batch load all CSV files matching the pattern from the data directory.

        Parameters:
            pattern: File name pattern (default '*.csv').
            validate: Whether to validate each file.
            silent: If True, suppresses console output.

        Returns:
            StockDataset containing all successfully loaded stocks.
        """
        csv_files = sorted(self.data_dir.glob(pattern))
        if not csv_files:
            raise FileNotFoundError(
                f"No files matching '{pattern}' found in directory: {self.data_dir}"
            )

        dataset = StockDataset()
        errors: dict[str, str] = {}

        for file_path in csv_files:
            try:
                stock = self.load_stock(file_path, validate=validate)
                dataset.add(stock)
            except Exception as err:
                errors[file_path.name] = str(err)

        if not silent:
            print(
                f"Successfully loaded {len(dataset)} stocks from directory: {self.data_dir}"
            )
            if errors:
                print(f"Warning: skipped {len(errors)} files with errors:")
                for fname, msg in errors.items():
                    print(f"  - {fname}: {msg}")

        return dataset


# ==============================================================================
# Module-level convenience helper functions
# ==============================================================================

def load_stock_data(
    source: Union[str, Path],
    data_dir: Optional[Union[str, Path]] = None,
    validate: bool = True,
) -> StockData:
    """Load data for a single stock (convenience module function)."""
    loader = DataLoader(data_dir=data_dir)
    return loader.load_stock(source=source, validate=validate)


def load_all_stocks(
    data_dir: Optional[Union[str, Path]] = None,
    pattern: str = "*.csv",
    validate: bool = True,
) -> StockDataset:
    """Batch load all stocks from data directory (convenience module function)."""
    loader = DataLoader(data_dir=data_dir)
    return loader.load_all(pattern=pattern, validate=validate)
