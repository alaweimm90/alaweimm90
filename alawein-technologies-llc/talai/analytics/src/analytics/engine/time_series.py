"""
Time Series Analysis for TalAI

Advanced time series analytics for validation trends, forecasting,
and pattern detection in research workflows.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy import signal, stats
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class TimeSeriesAnalyzer:
    """
    Advanced time series analysis for validation metrics.

    Features:
    - Trend detection and decomposition
    - Seasonality analysis
    - Forecasting with multiple models
    - Anomaly detection in time series
    - Change point detection
    - Cross-correlation analysis
    """

    def __init__(self):
        """Initialize time series analyzer."""
        self.scaler = StandardScaler()
        self.forecasting_models = {}

    async def analyze_validation_trends(self,
                                       data: List[Dict],
                                       metric_name: str = 'success_rate',
                                       frequency: str = 'D') -> Dict[str, Any]:
        """
        Analyze trends in validation metrics over time.

        Args:
            data: List of validation records with timestamps
            metric_name: Metric to analyze
            frequency: Time frequency for resampling

        Returns:
            Comprehensive trend analysis
        """
        try:
            # Convert to pandas DataFrame
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp').sort_index()

            # Resample to regular frequency
            ts = df[metric_name].resample(frequency).mean()
            ts = ts.fillna(method='ffill')  # Forward fill missing values

            # Decompose time series
            decomposition = await self._decompose_time_series(ts)

            # Detect trends
            trend_info = await self._detect_trends(ts)

            # Find seasonality
            seasonality = await self._analyze_seasonality(ts)

            # Detect anomalies
            anomalies = await self._detect_anomalies(ts)

            # Change point detection
            change_points = await self._detect_change_points(ts)

            # Generate forecast
            forecast = await self._generate_forecast(ts)

            return {
                'metric': metric_name,
                'frequency': frequency,
                'time_range': {
                    'start': ts.index[0].isoformat(),
                    'end': ts.index[-1].isoformat(),
                    'duration_days': (ts.index[-1] - ts.index[0]).days
                },
                'statistics': {
                    'mean': float(ts.mean()),
                    'std': float(ts.std()),
                    'min': float(ts.min()),
                    'max': float(ts.max()),
                    'median': float(ts.median()),
                    'skewness': float(stats.skew(ts)),
                    'kurtosis': float(stats.kurtosis(ts))
                },
                'decomposition': decomposition,
                'trend': trend_info,
                'seasonality': seasonality,
                'anomalies': anomalies,
                'change_points': change_points,
                'forecast': forecast
            }

        except Exception as e:
            logger.error(f"Error analyzing validation trends: {e}")
            return {'error': str(e)}

    async def _decompose_time_series(self, ts: pd.Series) -> Dict[str, Any]:
        """Decompose time series into trend, seasonal, and residual components."""
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose

            # Perform decomposition
            decomposition = seasonal_decompose(ts, model='multiplicative', period=7)

            return {
                'trend': decomposition.trend.dropna().tolist(),
                'seasonal': decomposition.seasonal.dropna().tolist(),
                'residual': decomposition.resid.dropna().tolist(),
                'trend_strength': float(1 - np.var(decomposition.resid) / np.var(ts)),
                'seasonal_strength': float(1 - np.var(decomposition.resid) /
                                         np.var(ts - decomposition.seasonal))
            }
        except Exception:
            # Fallback for insufficient data
            return {
                'trend': ts.tolist(),
                'seasonal': [0] * len(ts),
                'residual': [0] * len(ts),
                'trend_strength': 0.0,
                'seasonal_strength': 0.0
            }

    async def _detect_trends(self, ts: pd.Series) -> Dict[str, Any]:
        """Detect and quantify trends in time series."""
        try:
            # Linear trend
            x = np.arange(len(ts))
            y = ts.values
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

            # Polynomial trend (degree 2)
            poly_coef = np.polyfit(x, y, 2)
            poly_trend = np.polyval(poly_coef, x)
            poly_r2 = 1 - (np.sum((y - poly_trend) ** 2) / np.sum((y - np.mean(y)) ** 2))

            # Moving averages
            ma_7 = ts.rolling(window=7, min_periods=1).mean()
            ma_30 = ts.rolling(window=30, min_periods=1).mean()

            # Trend direction
            recent_slope = (ts.iloc[-1] - ts.iloc[-min(7, len(ts))]) / min(7, len(ts))

            return {
                'linear': {
                    'slope': float(slope),
                    'intercept': float(intercept),
                    'r_squared': float(r_value ** 2),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05
                },
                'polynomial': {
                    'coefficients': poly_coef.tolist(),
                    'r_squared': float(poly_r2)
                },
                'moving_averages': {
                    'ma_7_current': float(ma_7.iloc[-1]),
                    'ma_30_current': float(ma_30.iloc[-1]) if len(ts) >= 30 else None
                },
                'direction': 'increasing' if recent_slope > 0 else 'decreasing',
                'strength': abs(float(r_value)),
                'acceleration': float(poly_coef[0]) if len(poly_coef) > 0 else 0.0
            }
        except Exception as e:
            logger.error(f"Error detecting trends: {e}")
            return {'error': str(e)}

    async def _analyze_seasonality(self, ts: pd.Series) -> Dict[str, Any]:
        """Analyze seasonal patterns in time series."""
        try:
            # FFT for frequency analysis
            fft_values = np.fft.fft(ts.values)
            frequencies = np.fft.fftfreq(len(ts))
            power_spectrum = np.abs(fft_values) ** 2

            # Find dominant frequencies (exclude DC component)
            dominant_freq_idx = np.argmax(power_spectrum[1:len(power_spectrum)//2]) + 1
            dominant_period = 1 / frequencies[dominant_freq_idx] if frequencies[dominant_freq_idx] != 0 else len(ts)

            # Autocorrelation analysis
            acf_values = [ts.autocorr(lag=i) for i in range(min(30, len(ts)//2))]

            # Find seasonal period from ACF
            acf_peaks = signal.find_peaks(acf_values[1:], height=0.3)[0]
            seasonal_period = acf_peaks[0] + 1 if len(acf_peaks) > 0 else None

            # Day of week patterns
            if hasattr(ts.index, 'dayofweek'):
                dow_means = ts.groupby(ts.index.dayofweek).mean()
                dow_pattern = dow_means.to_dict()
            else:
                dow_pattern = {}

            return {
                'has_seasonality': seasonal_period is not None,
                'seasonal_period': int(seasonal_period) if seasonal_period else None,
                'dominant_frequency': float(frequencies[dominant_freq_idx]),
                'dominant_period_days': float(dominant_period),
                'autocorrelation': {
                    'lag_1': float(acf_values[1]) if len(acf_values) > 1 else 0.0,
                    'lag_7': float(acf_values[7]) if len(acf_values) > 7 else 0.0,
                    'max_correlation': float(max(acf_values[1:])) if len(acf_values) > 1 else 0.0
                },
                'day_of_week_pattern': dow_pattern,
                'seasonal_strength': float(max(acf_values[1:])) if len(acf_values) > 1 else 0.0
            }
        except Exception as e:
            logger.error(f"Error analyzing seasonality: {e}")
            return {'has_seasonality': False, 'error': str(e)}

    async def _detect_anomalies(self, ts: pd.Series) -> Dict[str, Any]:
        """Detect anomalies in time series using multiple methods."""
        try:
            anomalies = []

            # Method 1: Z-score
            z_scores = np.abs(stats.zscore(ts.values))
            z_threshold = 3
            z_anomalies = ts.index[z_scores > z_threshold].tolist()

            # Method 2: Isolation Forest
            from sklearn.ensemble import IsolationForest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            X = ts.values.reshape(-1, 1)
            predictions = iso_forest.fit_predict(X)
            iso_anomalies = ts.index[predictions == -1].tolist()

            # Method 3: Moving average deviation
            ma = ts.rolling(window=7, min_periods=1).mean()
            std = ts.rolling(window=7, min_periods=1).std()
            upper_bound = ma + 2 * std
            lower_bound = ma - 2 * std
            ma_anomalies = ts.index[(ts > upper_bound) | (ts < lower_bound)].tolist()

            # Combine anomalies (consensus approach)
            all_anomalies = set(z_anomalies) | set(iso_anomalies) | set(ma_anomalies)

            # Create anomaly details
            for idx in all_anomalies:
                if idx in ts.index:
                    anomalies.append({
                        'timestamp': idx.isoformat(),
                        'value': float(ts[idx]),
                        'z_score': float(z_scores[ts.index.get_loc(idx)]),
                        'methods_detected': []
                    })

                    if idx in z_anomalies:
                        anomalies[-1]['methods_detected'].append('z_score')
                    if idx in iso_anomalies:
                        anomalies[-1]['methods_detected'].append('isolation_forest')
                    if idx in ma_anomalies:
                        anomalies[-1]['methods_detected'].append('moving_average')

            # Sort by timestamp
            anomalies.sort(key=lambda x: x['timestamp'])

            return {
                'total_anomalies': len(anomalies),
                'anomaly_rate': len(anomalies) / len(ts) if len(ts) > 0 else 0.0,
                'anomalies': anomalies[:10],  # Return top 10
                'detection_methods': ['z_score', 'isolation_forest', 'moving_average']
            }

        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return {'total_anomalies': 0, 'error': str(e)}

    async def _detect_change_points(self, ts: pd.Series) -> Dict[str, Any]:
        """Detect change points in time series."""
        try:
            from ruptures import Pelt
            model = "rbf"  # Radial basis function

            # Prepare data
            signal_data = ts.values

            # Detect change points
            algo = Pelt(model=model, min_size=3, jump=1)
            algo.fit(signal_data)
            change_points = algo.predict(pen=np.log(len(signal_data)) * 2)

            # Remove the last point (which is always the end of the series)
            if change_points and change_points[-1] == len(signal_data):
                change_points = change_points[:-1]

            # Create change point details
            changes = []
            for cp_idx in change_points:
                if cp_idx < len(ts):
                    # Calculate magnitude of change
                    before_mean = ts.iloc[max(0, cp_idx-10):cp_idx].mean()
                    after_mean = ts.iloc[cp_idx:min(len(ts), cp_idx+10)].mean()
                    change_magnitude = after_mean - before_mean

                    changes.append({
                        'index': int(cp_idx),
                        'timestamp': ts.index[cp_idx].isoformat(),
                        'before_value': float(before_mean),
                        'after_value': float(after_mean),
                        'change_magnitude': float(change_magnitude),
                        'change_percent': float((change_magnitude / before_mean * 100)
                                              if before_mean != 0 else 0)
                    })

            return {
                'total_change_points': len(changes),
                'change_points': changes,
                'average_segment_length': len(ts) / (len(changes) + 1) if changes else len(ts),
                'stability_score': 1.0 / (1 + len(changes) / 10)  # More changes = less stability
            }

        except ImportError:
            # Fallback method using cumsum
            cumsum = np.cumsum(ts.values - np.mean(ts.values))
            change_points = signal.find_peaks(np.abs(np.diff(cumsum)), height=np.std(cumsum))[0]

            changes = []
            for cp_idx in change_points[:5]:  # Limit to top 5
                if cp_idx < len(ts) - 1:
                    changes.append({
                        'index': int(cp_idx),
                        'timestamp': ts.index[cp_idx].isoformat(),
                        'value': float(ts.iloc[cp_idx])
                    })

            return {
                'total_change_points': len(changes),
                'change_points': changes,
                'method': 'cumsum_fallback'
            }
        except Exception as e:
            logger.error(f"Error detecting change points: {e}")
            return {'total_change_points': 0, 'error': str(e)}

    async def _generate_forecast(self,
                                ts: pd.Series,
                                horizon: int = 7) -> Dict[str, Any]:
        """Generate forecasts using multiple methods."""
        try:
            forecasts = {}

            # Method 1: Simple Moving Average
            ma_window = min(7, len(ts) // 2)
            ma_forecast = [ts.rolling(window=ma_window).mean().iloc[-1]] * horizon
            forecasts['moving_average'] = ma_forecast

            # Method 2: Linear Regression
            x = np.arange(len(ts))
            y = ts.values
            slope, intercept, _, _, _ = stats.linregress(x, y)
            future_x = np.arange(len(ts), len(ts) + horizon)
            lr_forecast = slope * future_x + intercept
            forecasts['linear_regression'] = lr_forecast.tolist()

            # Method 3: Exponential Smoothing
            alpha = 0.3  # Smoothing parameter
            es_forecast = []
            last_value = ts.iloc[-1]
            for _ in range(horizon):
                es_forecast.append(last_value)
                last_value = alpha * last_value + (1 - alpha) * ts.mean()
            forecasts['exponential_smoothing'] = es_forecast

            # Method 4: ARIMA (simplified)
            try:
                from statsmodels.tsa.arima.model import ARIMA
                model = ARIMA(ts, order=(1, 1, 1))
                model_fit = model.fit(disp=False)
                arima_forecast = model_fit.forecast(steps=horizon)
                forecasts['arima'] = arima_forecast.tolist()
            except:
                # Fallback to persistence
                forecasts['arima'] = [ts.iloc[-1]] * horizon

            # Calculate ensemble forecast
            ensemble = np.mean([forecasts[method] for method in forecasts], axis=0)

            # Calculate prediction intervals
            forecast_std = np.std([forecasts[method] for method in forecasts], axis=0)
            lower_bound = ensemble - 1.96 * forecast_std
            upper_bound = ensemble + 1.96 * forecast_std

            return {
                'horizon_days': horizon,
                'methods': list(forecasts.keys()),
                'forecasts': forecasts,
                'ensemble_forecast': ensemble.tolist(),
                'prediction_interval': {
                    'lower': lower_bound.tolist(),
                    'upper': upper_bound.tolist(),
                    'confidence': 0.95
                },
                'forecast_dates': pd.date_range(
                    start=ts.index[-1] + pd.Timedelta(days=1),
                    periods=horizon,
                    freq='D'
                ).strftime('%Y-%m-%d').tolist()
            }

        except Exception as e:
            logger.error(f"Error generating forecast: {e}")
            return {'error': str(e)}

    async def cross_correlate_metrics(self,
                                     metric1: pd.Series,
                                     metric2: pd.Series,
                                     max_lag: int = 30) -> Dict[str, Any]:
        """
        Compute cross-correlation between two time series.

        Args:
            metric1: First time series
            metric2: Second time series
            max_lag: Maximum lag to consider

        Returns:
            Cross-correlation analysis
        """
        try:
            # Align series
            aligned = pd.DataFrame({'metric1': metric1, 'metric2': metric2}).dropna()

            if len(aligned) < 10:
                return {'error': 'Insufficient data for correlation analysis'}

            # Normalize series
            norm1 = (aligned['metric1'] - aligned['metric1'].mean()) / aligned['metric1'].std()
            norm2 = (aligned['metric2'] - aligned['metric2'].mean()) / aligned['metric2'].std()

            # Compute cross-correlation at different lags
            correlations = []
            for lag in range(-max_lag, max_lag + 1):
                if lag < 0:
                    corr = norm1.iloc[:lag].corr(norm2.iloc[-lag:])
                elif lag > 0:
                    corr = norm1.iloc[lag:].corr(norm2.iloc[:-lag])
                else:
                    corr = norm1.corr(norm2)

                correlations.append({
                    'lag': lag,
                    'correlation': float(corr) if not np.isnan(corr) else 0.0
                })

            # Find optimal lag
            max_corr = max(correlations, key=lambda x: abs(x['correlation']))

            # Granger causality test (simplified)
            granger_result = await self._granger_causality_test(
                aligned['metric1'].values,
                aligned['metric2'].values
            )

            return {
                'max_correlation': max_corr['correlation'],
                'optimal_lag': max_corr['lag'],
                'correlations': correlations,
                'simultaneous_correlation': float(norm1.corr(norm2)),
                'granger_causality': granger_result,
                'relationship_strength': 'strong' if abs(max_corr['correlation']) > 0.7
                                       else 'moderate' if abs(max_corr['correlation']) > 0.4
                                       else 'weak'
            }

        except Exception as e:
            logger.error(f"Error in cross-correlation: {e}")
            return {'error': str(e)}

    async def _granger_causality_test(self,
                                     x: np.ndarray,
                                     y: np.ndarray,
                                     max_lag: int = 2) -> Dict[str, Any]:
        """Simplified Granger causality test."""
        try:
            from sklearn.linear_model import LinearRegression

            # Model 1: Y predicted by its own lags
            X1 = np.column_stack([y[i:-(max_lag-i)] for i in range(1, max_lag+1)])
            y1 = y[max_lag:]
            model1 = LinearRegression().fit(X1, y1)
            rss1 = np.sum((y1 - model1.predict(X1)) ** 2)

            # Model 2: Y predicted by its own lags and X lags
            X2 = np.column_stack(
                [y[i:-(max_lag-i)] for i in range(1, max_lag+1)] +
                [x[i:-(max_lag-i)] for i in range(1, max_lag+1)]
            )
            model2 = LinearRegression().fit(X2, y1)
            rss2 = np.sum((y1 - model2.predict(X2)) ** 2)

            # F-statistic
            f_stat = ((rss1 - rss2) / max_lag) / (rss2 / (len(y1) - 2 * max_lag - 1))
            p_value = 1 - stats.f.cdf(f_stat, max_lag, len(y1) - 2 * max_lag - 1)

            return {
                'causes_y': p_value < 0.05,
                'f_statistic': float(f_stat),
                'p_value': float(p_value),
                'lag_order': max_lag
            }

        except Exception:
            return {'causes_y': False, 'error': 'Unable to compute Granger causality'}