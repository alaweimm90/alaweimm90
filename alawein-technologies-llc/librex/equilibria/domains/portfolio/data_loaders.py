"""
Portfolio Data Loaders

Utilities for loading financial data from various sources:
- CSV files
- Yahoo Finance (via yfinance)
- Synthetic data generation
- Data cleaning and preprocessing
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict, Union
from datetime import datetime, timedelta
import warnings


class DataLoader:
    """
    Unified data loader for portfolio optimization

    Handles various data sources and preprocessing
    """

    def __init__(self, frequency: str = 'daily'):
        """
        Initialize data loader

        Args:
            frequency: Data frequency ('daily', 'weekly', 'monthly')
        """
        self.frequency = frequency
        self.data_cache = {}

    def load_from_csv(self,
                     filepath: str,
                     date_column: str = 'Date',
                     price_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load price data from CSV file

        Args:
            filepath: Path to CSV file
            date_column: Name of date column
            price_columns: List of price column names (all numeric if None)

        Returns:
            DataFrame of prices (dates as index, assets as columns)
        """
        # Read CSV
        df = pd.read_csv(filepath)

        # Parse dates
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.set_index(date_column)

        # Select price columns
        if price_columns is None:
            # Use all numeric columns
            price_df = df.select_dtypes(include=[np.number])
        else:
            price_df = df[price_columns]

        # Sort by date
        price_df = price_df.sort_index()

        # Store in cache
        self.data_cache['prices'] = price_df

        return price_df

    def load_from_yahoo(self,
                       tickers: List[str],
                       start_date: Union[str, datetime],
                       end_date: Union[str, datetime],
                       price_type: str = 'Adj Close') -> pd.DataFrame:
        """
        Load price data from Yahoo Finance

        Args:
            tickers: List of stock tickers
            start_date: Start date for data
            end_date: End date for data
            price_type: Type of price ('Open', 'High', 'Low', 'Close', 'Adj Close')

        Returns:
            DataFrame of prices
        """
        try:
            import yfinance as yf
        except ImportError:
            warnings.warn("yfinance not installed, returning synthetic data")
            return self.generate_synthetic_prices(
                n_assets=len(tickers),
                n_periods=252,
                asset_names=tickers
            )

        # Download data
        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True
        )

        # Extract price type
        if len(tickers) == 1:
            prices = data[price_type].to_frame(tickers[0])
        else:
            prices = data[price_type]

        # Handle missing data
        prices = self.clean_price_data(prices)

        # Store in cache
        self.data_cache['prices'] = prices

        return prices

    def generate_synthetic_prices(self,
                                 n_assets: int = 10,
                                 n_periods: int = 1000,
                                 annual_return: float = 0.08,
                                 annual_volatility: float = 0.15,
                                 correlation: float = 0.3,
                                 seed: int = 42,
                                 asset_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Generate synthetic price data for testing

        Args:
            n_assets: Number of assets
            n_periods: Number of time periods
            annual_return: Average annual return
            annual_volatility: Average annual volatility
            correlation: Average pairwise correlation
            seed: Random seed
            asset_names: Optional asset names

        Returns:
            DataFrame of synthetic prices
        """
        np.random.seed(seed)

        # Convert annual parameters to period parameters
        periods_per_year = {'daily': 252, 'weekly': 52, 'monthly': 12}.get(self.frequency, 252)
        period_return = annual_return / periods_per_year
        period_vol = annual_volatility / np.sqrt(periods_per_year)

        # Generate correlation matrix
        corr_matrix = np.full((n_assets, n_assets), correlation)
        np.fill_diagonal(corr_matrix, 1.0)

        # Generate individual volatilities (some variation)
        vols = period_vol * (0.5 + 1.5 * np.random.random(n_assets))

        # Convert correlation to covariance
        cov_matrix = np.outer(vols, vols) * corr_matrix

        # Generate returns
        returns = np.random.multivariate_normal(
            mean=np.full(n_assets, period_return),
            cov=cov_matrix,
            size=n_periods
        )

        # Convert to prices
        prices = 100 * np.exp(np.cumsum(returns, axis=0))

        # Create DataFrame
        dates = pd.date_range(end=datetime.now(), periods=n_periods, freq={'daily': 'D', 'weekly': 'W', 'monthly': 'M'}.get(self.frequency, 'D'))

        if asset_names is None:
            asset_names = [f'Asset_{i+1}' for i in range(n_assets)]

        price_df = pd.DataFrame(prices, index=dates, columns=asset_names)

        # Store in cache
        self.data_cache['prices'] = price_df

        return price_df

    def prices_to_returns(self,
                         prices: pd.DataFrame,
                         method: str = 'simple',
                         dropna: bool = True) -> pd.DataFrame:
        """
        Convert prices to returns

        Args:
            prices: DataFrame of prices
            method: 'simple' or 'log' returns
            dropna: Whether to drop NaN values

        Returns:
            DataFrame of returns
        """
        if method == 'simple':
            returns = prices.pct_change()
        elif method == 'log':
            returns = np.log(prices / prices.shift(1))
        else:
            raise ValueError(f"Unknown method: {method}")

        if dropna:
            returns = returns.dropna()

        # Store in cache
        self.data_cache['returns'] = returns

        return returns

    def clean_price_data(self,
                        prices: pd.DataFrame,
                        method: str = 'forward_fill',
                        max_missing_pct: float = 0.1) -> pd.DataFrame:
        """
        Clean and preprocess price data

        Args:
            prices: DataFrame of prices
            method: Method for handling missing data ('forward_fill', 'interpolate', 'drop')
            max_missing_pct: Maximum percentage of missing data allowed per column

        Returns:
            Cleaned price DataFrame
        """
        # Check missing data percentage
        missing_pct = prices.isnull().sum() / len(prices)
        high_missing = missing_pct[missing_pct > max_missing_pct]

        if len(high_missing) > 0:
            warnings.warn(f"Columns with >{max_missing_pct:.1%} missing data: {high_missing.index.tolist()}")
            # Drop columns with too much missing data
            prices = prices.drop(columns=high_missing.index)

        # Handle remaining missing data
        if method == 'forward_fill':
            prices = prices.fillna(method='ffill').fillna(method='bfill')
        elif method == 'interpolate':
            prices = prices.interpolate(method='time')
        elif method == 'drop':
            prices = prices.dropna()
        else:
            raise ValueError(f"Unknown method: {method}")

        # Remove any remaining NaN rows
        prices = prices.dropna()

        return prices

    def load_factor_data(self,
                        factor_names: List[str] = ['MKT', 'SMB', 'HML'],
                        source: str = 'fama_french') -> pd.DataFrame:
        """
        Load factor return data

        Args:
            factor_names: Names of factors to load
            source: Data source ('fama_french', 'custom')

        Returns:
            DataFrame of factor returns
        """
        if source == 'fama_french':
            try:
                import pandas_datareader as pdr
                # Load Fama-French factors
                ff_factors = pdr.get_data_famafrench('F-F_Research_Data_Factors', start='2010')[0]
                ff_factors = ff_factors[factor_names] / 100  # Convert to decimal
                return ff_factors
            except:
                warnings.warn("Unable to load Fama-French data, using synthetic factors")

        # Generate synthetic factor data
        n_periods = 1000
        n_factors = len(factor_names)

        # Factor returns with different characteristics
        factor_returns = np.random.randn(n_periods, n_factors) * 0.01

        # Add some autocorrelation
        for i in range(1, n_periods):
            factor_returns[i] = 0.1 * factor_returns[i-1] + 0.9 * factor_returns[i]

        dates = pd.date_range(end=datetime.now(), periods=n_periods, freq='D')
        factor_df = pd.DataFrame(factor_returns, index=dates, columns=factor_names)

        return factor_df

    def prepare_portfolio_data(self,
                              tickers: Optional[List[str]] = None,
                              lookback_years: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for portfolio optimization

        Args:
            tickers: List of tickers (uses default if None)
            lookback_years: Years of historical data

        Returns:
            Tuple of (prices DataFrame, returns DataFrame)
        """
        if tickers is None:
            # Default portfolio of diversified ETFs
            tickers = ['SPY', 'AGG', 'GLD', 'VNQ', 'EFA', 'EEM', 'DBC', 'TLT']

        # Calculate dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_years * 365)

        # Load data
        prices = self.load_from_yahoo(tickers, start_date, end_date)

        # Calculate returns
        returns = self.prices_to_returns(prices)

        return prices, returns

    def calculate_statistics(self, returns: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate basic statistics for returns

        Args:
            returns: DataFrame of returns

        Returns:
            Dictionary of statistics
        """
        stats = {
            'mean': returns.mean(),
            'std': returns.std(),
            'skew': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'min': returns.min(),
            'max': returns.max(),
            'sharpe': returns.mean() / returns.std() * np.sqrt(252)
        }

        return stats

    def create_market_scenarios(self,
                               base_returns: pd.DataFrame,
                               n_scenarios: int = 1000,
                               scenario_types: List[str] = ['normal', 'crisis', 'boom']) -> np.ndarray:
        """
        Generate market scenarios for stress testing

        Args:
            base_returns: Historical returns for calibration
            n_scenarios: Number of scenarios to generate
            scenario_types: Types of scenarios to include

        Returns:
            Array of scenario returns (n_scenarios x n_assets)
        """
        n_assets = base_returns.shape[1]
        scenarios_per_type = n_scenarios // len(scenario_types)
        all_scenarios = []

        # Calculate base statistics
        mean = base_returns.mean().values
        cov = base_returns.cov().values

        for scenario_type in scenario_types:
            if scenario_type == 'normal':
                # Normal market conditions
                scenarios = np.random.multivariate_normal(mean, cov, scenarios_per_type)

            elif scenario_type == 'crisis':
                # Crisis: higher correlation, negative returns
                crisis_mean = mean - 2 * np.sqrt(np.diag(cov))
                crisis_corr = 0.8
                crisis_cov = np.sqrt(np.outer(np.diag(cov), np.diag(cov))) * crisis_corr
                np.fill_diagonal(crisis_cov, np.diag(cov))
                scenarios = np.random.multivariate_normal(crisis_mean, crisis_cov * 2, scenarios_per_type)

            elif scenario_type == 'boom':
                # Boom: lower correlation, positive returns
                boom_mean = mean + np.sqrt(np.diag(cov))
                boom_corr = 0.2
                boom_cov = np.sqrt(np.outer(np.diag(cov), np.diag(cov))) * boom_corr
                np.fill_diagonal(boom_cov, np.diag(cov))
                scenarios = np.random.multivariate_normal(boom_mean, boom_cov * 0.5, scenarios_per_type)

            else:
                # Default to normal
                scenarios = np.random.multivariate_normal(mean, cov, scenarios_per_type)

            all_scenarios.append(scenarios)

        return np.vstack(all_scenarios)

    def split_train_test(self,
                        data: pd.DataFrame,
                        test_size: float = 0.2,
                        gap: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and testing sets

        Args:
            data: DataFrame to split
            test_size: Fraction of data for testing
            gap: Number of periods gap between train and test

        Returns:
            Tuple of (train_data, test_data)
        """
        n = len(data)
        test_periods = int(n * test_size)
        train_end = n - test_periods - gap

        train_data = data.iloc[:train_end]
        test_data = data.iloc[train_end + gap:]

        return train_data, test_data


# Utility functions for quick data loading

def load_sample_portfolio_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load sample portfolio data for testing

    Returns:
        Tuple of (prices, returns) DataFrames
    """
    loader = DataLoader()
    prices, returns = loader.prepare_portfolio_data()
    return prices, returns


def generate_test_data(n_assets: int = 5,
                      n_periods: int = 1000) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate test data for portfolio optimization

    Args:
        n_assets: Number of assets
        n_periods: Number of periods

    Returns:
        Tuple of (prices, returns) DataFrames
    """
    loader = DataLoader()
    prices = loader.generate_synthetic_prices(n_assets, n_periods)
    returns = loader.prices_to_returns(prices)
    return prices, returns