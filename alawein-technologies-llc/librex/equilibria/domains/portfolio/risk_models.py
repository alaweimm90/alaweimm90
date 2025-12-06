"""
Portfolio Risk Models and Covariance Estimators

Implements various methods for estimating expected returns and covariance matrices:
- Historical estimation
- Shrinkage estimators (Ledoit-Wolf)
- Exponentially weighted estimation
- Factor models
- Robust covariance estimation
- GARCH models
"""

import numpy as np
from typing import Optional, Tuple, Dict, Union
import warnings
from scipy import stats
from scipy.linalg import sqrtm
from sklearn.covariance import LedoitWolf, MinCovDet, EmpiricalCovariance, OAS


class RiskEstimator:
    """
    Advanced risk estimation methods for portfolio optimization

    Provides various covariance matrix estimators and return predictors
    """

    def __init__(self, returns: np.ndarray, frequency: str = 'daily'):
        """
        Initialize risk estimator

        Args:
            returns: Historical returns matrix (T x N)
            frequency: Return frequency ('daily', 'weekly', 'monthly')
        """
        self.returns = returns
        self.n_periods, self.n_assets = returns.shape
        self.frequency = frequency

        # Annualization factors
        self.annualization_factors = {
            'daily': 252,
            'weekly': 52,
            'monthly': 12,
            'quarterly': 4,
            'yearly': 1
        }
        self.periods_per_year = self.annualization_factors.get(frequency, 252)

    # ===== Return Estimation Methods =====

    def historical_mean(self) -> np.ndarray:
        """
        Simple historical mean returns

        Returns:
            Expected return vector
        """
        return np.mean(self.returns, axis=0)

    def exponentially_weighted_mean(self, halflife: int = 60) -> np.ndarray:
        """
        Exponentially weighted mean returns

        Args:
            halflife: Halflife for exponential decay (in periods)

        Returns:
            Expected return vector
        """
        decay = 0.5 ** (1 / halflife)
        weights = decay ** np.arange(self.n_periods - 1, -1, -1)
        weights /= weights.sum()

        return np.average(self.returns, axis=0, weights=weights)

    def james_stein_mean(self) -> np.ndarray:
        """
        James-Stein shrinkage estimator for expected returns

        Shrinks individual asset means toward grand mean

        Returns:
            Shrunk expected return vector
        """
        sample_mean = self.historical_mean()
        grand_mean = np.mean(sample_mean)

        # Shrinkage intensity
        sample_cov = np.cov(self.returns.T)
        var_sample_mean = np.diag(sample_cov) / self.n_periods

        # James-Stein shrinkage factor
        numerator = np.sum(var_sample_mean)
        denominator = np.sum((sample_mean - grand_mean) ** 2)

        if denominator < 1e-10:
            shrinkage = 1.0
        else:
            shrinkage = min(1.0, numerator / denominator)

        # Shrink toward grand mean
        shrunk_mean = (1 - shrinkage) * sample_mean + shrinkage * grand_mean

        return shrunk_mean

    def capm_expected_returns(self,
                             market_returns: np.ndarray,
                             risk_free_rate: float = 0.0) -> np.ndarray:
        """
        CAPM-based expected returns

        E[R_i] = R_f + β_i * (E[R_m] - R_f)

        Args:
            market_returns: Market portfolio returns
            risk_free_rate: Risk-free rate (annualized)

        Returns:
            Expected return vector
        """
        # Convert annual risk-free rate to period rate
        rf_period = risk_free_rate / self.periods_per_year

        # Calculate betas
        market_excess = market_returns - rf_period
        market_var = np.var(market_excess)

        betas = np.zeros(self.n_assets)
        for i in range(self.n_assets):
            asset_excess = self.returns[:, i] - rf_period
            betas[i] = np.cov(asset_excess, market_excess)[0, 1] / market_var

        # Market risk premium
        market_premium = np.mean(market_excess)

        # CAPM expected returns
        expected_returns = rf_period + betas * market_premium

        return expected_returns

    # ===== Covariance Estimation Methods =====

    def sample_covariance(self, annualized: bool = False) -> np.ndarray:
        """
        Simple sample covariance matrix

        Args:
            annualized: If True, annualize the covariance

        Returns:
            Covariance matrix
        """
        cov = np.cov(self.returns.T)

        if annualized:
            cov *= self.periods_per_year

        return cov

    def ledoit_wolf_shrinkage(self,
                             annualized: bool = False,
                             shrinkage: Optional[float] = None) -> Tuple[np.ndarray, float]:
        """
        Ledoit-Wolf shrinkage estimator

        Shrinks sample covariance toward structured estimator

        Args:
            annualized: If True, annualize the covariance
            shrinkage: Manual shrinkage parameter (auto if None)

        Returns:
            Tuple of (shrunk covariance matrix, shrinkage parameter)
        """
        lw = LedoitWolf(store_precision=False, assume_centered=False)

        if shrinkage is not None:
            # Manual shrinkage
            sample_cov = self.sample_covariance()
            # Target: diagonal matrix with sample variances
            target = np.diag(np.diag(sample_cov))
            shrunk_cov = (1 - shrinkage) * sample_cov + shrinkage * target
            actual_shrinkage = shrinkage
        else:
            # Automatic shrinkage
            shrunk_cov = lw.fit(self.returns).covariance_
            actual_shrinkage = lw.shrinkage_

        if annualized:
            shrunk_cov *= self.periods_per_year

        return shrunk_cov, actual_shrinkage

    def oracle_approximating_shrinkage(self, annualized: bool = False) -> np.ndarray:
        """
        Oracle Approximating Shrinkage (OAS) estimator

        Improved shrinkage method assuming Gaussian distribution

        Args:
            annualized: If True, annualize the covariance

        Returns:
            Shrunk covariance matrix
        """
        oas = OAS(store_precision=False, assume_centered=False)
        shrunk_cov = oas.fit(self.returns).covariance_

        if annualized:
            shrunk_cov *= self.periods_per_year

        return shrunk_cov

    def exponentially_weighted_covariance(self,
                                         halflife: int = 60,
                                         annualized: bool = False) -> np.ndarray:
        """
        Exponentially weighted covariance matrix

        Recent observations get higher weights

        Args:
            halflife: Halflife for exponential decay (in periods)
            annualized: If True, annualize the covariance

        Returns:
            Weighted covariance matrix
        """
        # Calculate weights
        decay = 0.5 ** (1 / halflife)
        weights = decay ** np.arange(self.n_periods - 1, -1, -1)
        weights /= weights.sum()

        # Weighted mean
        mean = np.average(self.returns, axis=0, weights=weights)

        # Weighted covariance
        centered = self.returns - mean
        weighted_cov = np.zeros((self.n_assets, self.n_assets))

        for t in range(self.n_periods):
            weighted_cov += weights[t] * np.outer(centered[t], centered[t])

        if annualized:
            weighted_cov *= self.periods_per_year

        return weighted_cov

    def minimum_covariance_determinant(self,
                                      support_fraction: float = 0.75,
                                      annualized: bool = False) -> np.ndarray:
        """
        Minimum Covariance Determinant (MCD) robust estimator

        Robust to outliers

        Args:
            support_fraction: Fraction of points to use for MCD
            annualized: If True, annualize the covariance

        Returns:
            Robust covariance matrix
        """
        mcd = MinCovDet(support_fraction=support_fraction, random_state=42)
        robust_cov = mcd.fit(self.returns).covariance_

        if annualized:
            robust_cov *= self.periods_per_year

        return robust_cov

    def factor_model_covariance(self,
                               factor_returns: np.ndarray,
                               method: str = 'pca') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Factor model based covariance estimation

        Σ = B * F * B' + D

        Where:
        - B is factor loadings matrix
        - F is factor covariance
        - D is diagonal matrix of idiosyncratic variances

        Args:
            factor_returns: Factor return time series (T x K)
            method: 'regression' or 'pca'

        Returns:
            Tuple of (covariance matrix, factor loadings, idiosyncratic risk)
        """
        n_factors = factor_returns.shape[1]

        if method == 'regression':
            # Regression-based factor loadings
            loadings = np.zeros((self.n_assets, n_factors))
            idio_var = np.zeros(self.n_assets)

            for i in range(self.n_assets):
                # Regress asset returns on factors
                X = np.column_stack([np.ones(self.n_periods), factor_returns])
                y = self.returns[:, i]
                beta = np.linalg.lstsq(X, y, rcond=None)[0]

                loadings[i] = beta[1:]  # Exclude intercept
                residuals = y - X @ beta
                idio_var[i] = np.var(residuals)

        elif method == 'pca':
            # PCA-based factor model
            from sklearn.decomposition import PCA

            pca = PCA(n_components=n_factors)
            pca.fit(self.returns)

            loadings = pca.components_.T
            transformed = pca.transform(self.returns)
            residuals = self.returns - pca.inverse_transform(transformed)
            idio_var = np.var(residuals, axis=0)

        else:
            raise ValueError(f"Unknown method: {method}")

        # Factor covariance
        factor_cov = np.cov(factor_returns.T)

        # Total covariance
        systematic_risk = loadings @ factor_cov @ loadings.T
        idiosyncratic_risk = np.diag(idio_var)
        total_cov = systematic_risk + idiosyncratic_risk

        return total_cov, loadings, np.sqrt(idio_var)

    def constant_correlation_model(self,
                                  correlation: Optional[float] = None,
                                  annualized: bool = False) -> np.ndarray:
        """
        Constant Correlation Model

        All pairwise correlations set to a constant

        Args:
            correlation: Constant correlation (average if None)
            annualized: If True, annualize the covariance

        Returns:
            Covariance matrix with constant correlations
        """
        sample_cov = self.sample_covariance()
        std_dev = np.sqrt(np.diag(sample_cov))

        if correlation is None:
            # Use average pairwise correlation
            sample_corr = np.corrcoef(self.returns.T)
            mask = np.triu(np.ones_like(sample_corr), k=1).astype(bool)
            correlation = np.mean(sample_corr[mask])

        # Constant correlation matrix
        corr_matrix = np.full((self.n_assets, self.n_assets), correlation)
        np.fill_diagonal(corr_matrix, 1.0)

        # Convert to covariance
        cov_matrix = np.outer(std_dev, std_dev) * corr_matrix

        if annualized:
            cov_matrix *= self.periods_per_year

        return cov_matrix

    def semi_covariance(self,
                       threshold: float = 0.0,
                       annualized: bool = False) -> np.ndarray:
        """
        Semi-covariance matrix (downside covariance)

        Only considers returns below threshold

        Args:
            threshold: Threshold for downside returns
            annualized: If True, annualize the covariance

        Returns:
            Semi-covariance matrix
        """
        # Filter downside returns
        downside_mask = np.any(self.returns < threshold, axis=1)
        downside_returns = self.returns[downside_mask]

        if len(downside_returns) < 2:
            warnings.warn("Insufficient downside observations, using full covariance")
            return self.sample_covariance(annualized)

        semi_cov = np.cov(downside_returns.T)

        if annualized:
            semi_cov *= self.periods_per_year

        return semi_cov

    def denoised_covariance(self,
                           method: str = 'marchenko_pastur',
                           annualized: bool = False) -> np.ndarray:
        """
        Denoised covariance using Random Matrix Theory

        Args:
            method: 'marchenko_pastur' or 'optimal_shrinkage'
            annualized: If True, annualize the covariance

        Returns:
            Denoised covariance matrix
        """
        sample_cov = self.sample_covariance()

        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(sample_cov)

        if method == 'marchenko_pastur':
            # Marchenko-Pastur distribution
            q = self.n_periods / self.n_assets

            if q > 1:
                lambda_plus = (1 + 1/np.sqrt(q)) ** 2
                lambda_minus = (1 - 1/np.sqrt(q)) ** 2

                # Identify noise eigenvalues
                lambda_median = np.median(eigenvalues)
                noise_threshold = lambda_median * lambda_plus

                # Denoise
                eigenvalues[eigenvalues < noise_threshold] = np.mean(
                    eigenvalues[eigenvalues < noise_threshold]
                )

        elif method == 'optimal_shrinkage':
            # Optimal shrinkage of eigenvalues
            # Following Ledoit-Wolf optimal nonlinear shrinkage
            n = self.n_assets
            sample_eigenvalues = eigenvalues * self.n_periods

            # Shrink eigenvalues
            c = n / self.n_periods
            lambda_star = eigenvalues.copy()

            for i in range(n):
                lambda_i = sample_eigenvalues[i]
                tmp = lambda_i / (1 - c + c * lambda_i)
                lambda_star[i] = tmp / self.n_periods

            eigenvalues = lambda_star

        else:
            raise ValueError(f"Unknown method: {method}")

        # Reconstruct covariance
        denoised_cov = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

        if annualized:
            denoised_cov *= self.periods_per_year

        return denoised_cov

    # ===== Combined Risk Estimator =====

    def estimate_risk(self,
                      method: str = 'ledoit_wolf',
                      return_method: str = 'historical',
                      **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate both expected returns and covariance matrix

        Args:
            method: Covariance estimation method
            return_method: Return estimation method
            **kwargs: Additional parameters for specific methods

        Returns:
            Tuple of (expected returns, covariance matrix)
        """
        # Estimate returns
        if return_method == 'historical':
            expected_returns = self.historical_mean()
        elif return_method == 'exponential':
            expected_returns = self.exponentially_weighted_mean(
                kwargs.get('return_halflife', 60)
            )
        elif return_method == 'james_stein':
            expected_returns = self.james_stein_mean()
        else:
            expected_returns = self.historical_mean()

        # Estimate covariance
        if method == 'sample':
            cov_matrix = self.sample_covariance()
        elif method == 'ledoit_wolf':
            cov_matrix, _ = self.ledoit_wolf_shrinkage()
        elif method == 'oas':
            cov_matrix = self.oracle_approximating_shrinkage()
        elif method == 'exponential':
            cov_matrix = self.exponentially_weighted_covariance(
                kwargs.get('cov_halflife', 60)
            )
        elif method == 'mcd':
            cov_matrix = self.minimum_covariance_determinant()
        elif method == 'constant_correlation':
            cov_matrix = self.constant_correlation_model()
        elif method == 'semi':
            cov_matrix = self.semi_covariance()
        elif method == 'denoised':
            cov_matrix = self.denoised_covariance()
        else:
            cov_matrix = self.sample_covariance()

        return expected_returns, cov_matrix

    def covariance_spectrum_analysis(self) -> Dict[str, Union[float, np.ndarray]]:
        """
        Analyze eigenvalue spectrum of covariance matrix

        Useful for understanding risk structure

        Returns:
            Dictionary with spectrum analysis
        """
        cov = self.sample_covariance()
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Variance explained
        total_var = eigenvalues.sum()
        var_explained = eigenvalues / total_var
        cumsum_var = np.cumsum(var_explained)

        # Effective rank (number of eigenvalues explaining 90% variance)
        effective_rank = np.argmax(cumsum_var >= 0.9) + 1

        # Condition number (numerical stability indicator)
        condition_number = eigenvalues[0] / eigenvalues[-1] if eigenvalues[-1] > 1e-10 else np.inf

        return {
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'variance_explained': var_explained,
            'cumulative_variance': cumsum_var,
            'effective_rank': effective_rank,
            'condition_number': condition_number,
            'is_well_conditioned': condition_number < 1000
        }