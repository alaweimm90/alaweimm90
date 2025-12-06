#!/usr/bin/env python3
"""
Performance History Tracking for Librex Benchmarks

This module provides functionality to track performance metrics over time,
detect regressions, and analyze trends across benchmark runs.
"""

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
import warnings


class BenchmarkHistory:
    """Tracks and analyzes benchmark performance over time"""

    def __init__(self, db_path: str = None):
        """Initialize the performance history tracker

        Args:
            db_path: Path to SQLite database (created if doesn't exist)
        """
        if db_path is None:
            db_path = Path(__file__).parent / "benchmark_results" / "history.db"
        else:
            db_path = Path(db_path)

        db_path.parent.mkdir(exist_ok=True, parents=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row  # Enable column access by name
        self._initialize_database()

    def _initialize_database(self):
        """Create database tables if they don't exist"""
        cursor = self.conn.cursor()

        # Main results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS benchmark_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                suite_name TEXT NOT NULL,
                git_commit TEXT,
                run_metadata TEXT
            )
        """)

        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON benchmark_runs (timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_suite ON benchmark_runs (suite_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_commit ON benchmark_runs (git_commit)")

        # Individual benchmark results
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS benchmark_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                problem_type TEXT NOT NULL,
                problem_size INTEGER,
                problem_instance TEXT,
                method TEXT NOT NULL,
                config TEXT,
                mean_objective REAL,
                std_objective REAL,
                best_objective REAL,
                worst_objective REAL,
                mean_runtime REAL,
                mean_memory REAL,
                optimality_gap REAL,
                success_rate REAL,
                convergence_speed INTEGER,
                FOREIGN KEY (run_id) REFERENCES benchmark_runs(id)
            )
        """)

        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_run_id ON benchmark_results (run_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_method ON benchmark_results (method)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_problem ON benchmark_results (problem_type, problem_size)")

        # Performance baselines
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_baselines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                suite_name TEXT NOT NULL,
                problem_key TEXT NOT NULL,
                method TEXT NOT NULL,
                baseline_objective REAL,
                baseline_runtime REAL,
                baseline_memory REAL,
                sample_size INTEGER,
                confidence_level REAL DEFAULT 0.95,
                UNIQUE(suite_name, problem_key, method)
            )
        """)

        # Regression alerts
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS regression_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                run_id INTEGER,
                suite_name TEXT,
                problem_key TEXT,
                method TEXT,
                metric TEXT,
                baseline_value REAL,
                current_value REAL,
                degradation_percent REAL,
                confidence REAL,
                alert_level TEXT,  -- 'warning', 'critical'
                resolved BOOLEAN DEFAULT FALSE,
                FOREIGN KEY (run_id) REFERENCES benchmark_runs(id)
            )
        """)

        self.conn.commit()

    def record_run(self, results: Dict) -> int:
        """Record a benchmark run in the database

        Args:
            results: Complete benchmark results dictionary

        Returns:
            Run ID for the recorded benchmark
        """
        cursor = self.conn.cursor()

        # Insert main run record
        cursor.execute("""
            INSERT INTO benchmark_runs (timestamp, suite_name, git_commit, run_metadata)
            VALUES (?, ?, ?, ?)
        """, (
            results.get('start_time', datetime.now().isoformat()),
            results.get('suite_name', 'unknown'),
            results.get('metadata', {}).get('git_commit'),
            json.dumps(results.get('metadata', {}))
        ))

        run_id = cursor.lastrowid

        # Insert individual benchmark results
        for result in results.get('results', []):
            if result.get('status') == 'failed':
                continue

            problem_info = result['problem']
            stats = result['statistics']

            problem_key = self._get_problem_key(problem_info)

            cursor.execute("""
                INSERT INTO benchmark_results (
                    run_id, problem_type, problem_size, problem_instance,
                    method, config,
                    mean_objective, std_objective, best_objective, worst_objective,
                    mean_runtime, mean_memory, optimality_gap, success_rate,
                    convergence_speed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id,
                problem_info.get('type'),
                problem_info.get('size'),
                problem_info.get('instance', ''),
                result['method'],
                json.dumps(result.get('config', {})),
                stats.get('mean_objective'),
                stats.get('std_objective'),
                stats.get('best_objective'),
                stats.get('worst_objective'),
                stats.get('mean_runtime'),
                stats.get('mean_memory'),
                stats.get('optimality_gap'),
                stats.get('success_rate'),
                result.get('convergence', {}).get('mean_iterations_to_90')
            ))

        self.conn.commit()
        return run_id

    def _get_problem_key(self, problem_info: Dict) -> str:
        """Generate a unique key for a problem configuration"""
        if 'instance' in problem_info:
            return f"{problem_info['type']}_{problem_info['instance']}"
        else:
            return f"{problem_info['type']}_size{problem_info.get('size', 0)}_inst{problem_info.get('instance_idx', 0)}"

    def update_baselines(
        self,
        suite_name: str,
        window_days: int = 7,
        confidence_level: float = 0.95
    ):
        """Update performance baselines based on recent runs

        Args:
            suite_name: Name of the benchmark suite
            window_days: Number of days to consider for baseline
            confidence_level: Confidence level for statistical tests
        """
        cursor = self.conn.cursor()

        # Get recent runs within the window
        cutoff_date = datetime.now() - timedelta(days=window_days)

        cursor.execute("""
            SELECT br.*, r.*
            FROM benchmark_results r
            JOIN benchmark_runs br ON r.run_id = br.id
            WHERE br.suite_name = ? AND br.timestamp >= ?
        """, (suite_name, cutoff_date.isoformat()))

        results = cursor.fetchall()

        # Group by problem and method
        from collections import defaultdict
        grouped = defaultdict(list)

        for row in results:
            # Handle problem key generation
            if row['problem_instance']:
                problem_key = f"{row['problem_type']}_{row['problem_instance']}"
            else:
                problem_key = f"{row['problem_type']}_size{row['problem_size']}"
            method = row['method']
            key = (problem_key, method)

            grouped[key].append({
                'objective': row['best_objective'],
                'runtime': row['mean_runtime'],
                'memory': row['mean_memory']
            })

        # Calculate baselines
        for (problem_key, method), measurements in grouped.items():
            if len(measurements) < 3:  # Need minimum samples
                continue

            objectives = [m['objective'] for m in measurements if m['objective'] is not None]
            runtimes = [m['runtime'] for m in measurements if m['runtime'] is not None]
            memories = [m['memory'] for m in measurements if m['memory'] is not None]

            # Calculate baseline values (using median for robustness)
            baseline_obj = np.median(objectives) if objectives else None
            baseline_rt = np.median(runtimes) if runtimes else None
            baseline_mem = np.median(memories) if memories else None

            # Update or insert baseline
            cursor.execute("""
                INSERT OR REPLACE INTO performance_baselines
                (suite_name, problem_key, method,
                 baseline_objective, baseline_runtime, baseline_memory,
                 sample_size, confidence_level)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                suite_name, problem_key, method,
                baseline_obj, baseline_rt, baseline_mem,
                len(measurements), confidence_level
            ))

        self.conn.commit()

    def detect_regressions(
        self,
        run_id: int,
        thresholds: Dict[str, float] = None
    ) -> List[Dict]:
        """Detect performance regressions in a benchmark run

        Args:
            run_id: ID of the benchmark run to analyze
            thresholds: Regression thresholds by metric

        Returns:
            List of detected regressions
        """
        if thresholds is None:
            thresholds = {
                'solution_quality': 0.05,  # 5% degradation
                'runtime': 0.20,  # 20% slower
                'memory_usage': 0.30  # 30% more memory
            }

        cursor = self.conn.cursor()

        # Get run info
        cursor.execute("""
            SELECT * FROM benchmark_runs WHERE id = ?
        """, (run_id,))
        run_info = cursor.fetchone()

        if not run_info:
            return []

        suite_name = run_info['suite_name']

        # Get results for this run
        cursor.execute("""
            SELECT * FROM benchmark_results WHERE run_id = ?
        """, (run_id,))
        results = cursor.fetchall()

        regressions = []

        for result in results:
            # Handle problem key generation
            if result['problem_instance']:
                problem_key = f"{result['problem_type']}_{result['problem_instance']}"
            else:
                problem_key = f"{result['problem_type']}_size{result['problem_size']}"
            method = result['method']

            # Get baseline
            cursor.execute("""
                SELECT * FROM performance_baselines
                WHERE suite_name = ? AND problem_key = ? AND method = ?
            """, (suite_name, problem_key, method))
            baseline = cursor.fetchone()

            if not baseline:
                continue

            # Check for regressions
            checks = [
                ('solution_quality', result['best_objective'], baseline['baseline_objective']),
                ('runtime', result['mean_runtime'], baseline['baseline_runtime']),
                ('memory_usage', result['mean_memory'], baseline['baseline_memory'])
            ]

            for metric, current_value, baseline_value in checks:
                if current_value is None or baseline_value is None:
                    continue

                # Calculate degradation
                if metric == 'solution_quality':
                    # For minimization problems, higher is worse
                    degradation = (current_value - baseline_value) / baseline_value
                else:
                    # For time/memory, higher is worse
                    degradation = (current_value - baseline_value) / baseline_value

                if degradation > thresholds.get(metric, 0.1):
                    # Statistical significance test
                    confidence = self._calculate_confidence(
                        metric, current_value, baseline_value, baseline['sample_size']
                    )

                    if confidence > 0.95:  # 95% confidence threshold
                        alert_level = 'critical' if degradation > thresholds[metric] * 2 else 'warning'

                        regression = {
                            'run_id': run_id,
                            'suite_name': suite_name,
                            'problem_key': problem_key,
                            'method': method,
                            'metric': metric,
                            'baseline_value': baseline_value,
                            'current_value': current_value,
                            'degradation_percent': degradation * 100,
                            'confidence': confidence,
                            'alert_level': alert_level
                        }

                        regressions.append(regression)

                        # Record alert in database
                        cursor.execute("""
                            INSERT INTO regression_alerts
                            (run_id, suite_name, problem_key, method, metric,
                             baseline_value, current_value, degradation_percent,
                             confidence, alert_level)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            run_id, suite_name, problem_key, method, metric,
                            baseline_value, current_value, degradation * 100,
                            confidence, alert_level
                        ))

        self.conn.commit()
        return regressions

    def _calculate_confidence(
        self,
        metric: str,
        current: float,
        baseline: float,
        sample_size: int
    ) -> float:
        """Calculate statistical confidence of a regression

        Uses a simplified approach based on z-scores
        """
        if sample_size < 2:
            return 0.0

        # Estimate standard error (simplified)
        # In practice, you'd want to use actual historical variance
        std_error = baseline * 0.1 / np.sqrt(sample_size)

        # Calculate z-score
        z_score = abs(current - baseline) / std_error

        # Convert to confidence (two-tailed test)
        confidence = 2 * stats.norm.cdf(z_score) - 1

        return min(1.0, max(0.0, confidence))

    def get_trend_analysis(
        self,
        suite_name: str,
        method: str,
        problem_key: str,
        days: int = 30
    ) -> Dict:
        """Analyze performance trends over time

        Args:
            suite_name: Benchmark suite name
            method: Optimization method
            problem_key: Problem identifier
            days: Number of days to analyze

        Returns:
            Dictionary with trend analysis results
        """
        cursor = self.conn.cursor()

        cutoff_date = datetime.now() - timedelta(days=days)

        cursor.execute("""
            SELECT br.timestamp, r.*
            FROM benchmark_results r
            JOIN benchmark_runs br ON r.run_id = br.id
            WHERE br.suite_name = ?
                AND r.method = ?
                AND br.timestamp >= ?
                AND (r.problem_instance = ? OR
                     (r.problem_type || '_size' || r.problem_size) = ?)
            ORDER BY br.timestamp
        """, (suite_name, method, cutoff_date.isoformat(), problem_key, problem_key))

        results = cursor.fetchall()

        if len(results) < 2:
            return {'status': 'insufficient_data'}

        # Extract time series
        timestamps = [r['timestamp'] for r in results]
        objectives = [r['best_objective'] for r in results]
        runtimes = [r['mean_runtime'] for r in results]

        # Convert timestamps to days from start
        start_date = pd.Timestamp(timestamps[0])
        days_from_start = [(pd.Timestamp(t) - start_date).days for t in timestamps]

        # Perform linear regression for trend
        from scipy.stats import linregress

        trends = {}

        if all(o is not None for o in objectives):
            slope, intercept, r_value, p_value, std_err = linregress(days_from_start, objectives)
            trends['objective'] = {
                'slope': slope,
                'slope_percent_per_day': (slope / np.mean(objectives)) * 100,
                'r_squared': r_value ** 2,
                'p_value': p_value,
                'trend': 'improving' if slope < 0 else 'degrading' if slope > 0 else 'stable'
            }

        if all(r is not None for r in runtimes):
            slope, intercept, r_value, p_value, std_err = linregress(days_from_start, runtimes)
            trends['runtime'] = {
                'slope': slope,
                'slope_percent_per_day': (slope / np.mean(runtimes)) * 100,
                'r_squared': r_value ** 2,
                'p_value': p_value,
                'trend': 'improving' if slope < 0 else 'degrading' if slope > 0 else 'stable'
            }

        return {
            'status': 'success',
            'data_points': len(results),
            'date_range': {
                'start': timestamps[0],
                'end': timestamps[-1]
            },
            'trends': trends,
            'latest_values': {
                'objective': objectives[-1],
                'runtime': runtimes[-1]
            }
        }

    def generate_comparison_matrix(
        self,
        suite_name: str,
        problem_type: str = None
    ) -> pd.DataFrame:
        """Generate a method comparison matrix

        Args:
            suite_name: Benchmark suite name
            problem_type: Optional filter by problem type

        Returns:
            DataFrame with method comparisons
        """
        cursor = self.conn.cursor()

        query = """
            SELECT
                r.problem_type,
                r.problem_size,
                r.problem_instance,
                r.method,
                AVG(r.best_objective) as avg_objective,
                AVG(r.mean_runtime) as avg_runtime,
                COUNT(*) as sample_count
            FROM benchmark_results r
            JOIN benchmark_runs br ON r.run_id = br.id
            WHERE br.suite_name = ?
        """
        params = [suite_name]

        if problem_type:
            query += " AND r.problem_type = ?"
            params.append(problem_type)

        query += """
            GROUP BY r.problem_type, r.problem_size, r.problem_instance, r.method
            ORDER BY r.problem_type, r.problem_size, r.problem_instance, avg_objective
        """

        cursor.execute(query, params)
        results = cursor.fetchall()

        # Convert to DataFrame
        df = pd.DataFrame(results, columns=[
            'problem_type', 'problem_size', 'problem_instance',
            'method', 'avg_objective', 'avg_runtime', 'sample_count'
        ])

        # Create pivot table for easy comparison
        if not df.empty:
            pivot = df.pivot_table(
                values='avg_objective',
                index=['problem_type', 'problem_size', 'problem_instance'],
                columns='method'
            )
            return pivot

        return pd.DataFrame()

    def cleanup_old_data(self, retention_days: int = 90):
        """Remove old benchmark data beyond retention period

        Args:
            retention_days: Number of days to retain data
        """
        cursor = self.conn.cursor()

        cutoff_date = datetime.now() - timedelta(days=retention_days)

        # Delete old runs and cascade to results
        cursor.execute("""
            DELETE FROM benchmark_runs
            WHERE timestamp < ?
        """, (cutoff_date.isoformat(),))

        # Delete old resolved alerts
        cursor.execute("""
            DELETE FROM regression_alerts
            WHERE detected_at < ? AND resolved = TRUE
        """, (cutoff_date.isoformat(),))

        self.conn.commit()

        # Vacuum database to reclaim space
        cursor.execute("VACUUM")

    def export_to_csv(self, output_dir: Path, suite_name: str = None):
        """Export benchmark history to CSV files

        Args:
            output_dir: Directory to save CSV files
            suite_name: Optional filter by suite name
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        cursor = self.conn.cursor()

        # Export main results
        query = """
            SELECT * FROM benchmark_results r
            JOIN benchmark_runs br ON r.run_id = br.id
        """
        if suite_name:
            query += f" WHERE br.suite_name = '{suite_name}'"

        df = pd.read_sql_query(query, self.conn)
        df.to_csv(output_dir / "benchmark_results.csv", index=False)

        # Export baselines
        df = pd.read_sql_query("SELECT * FROM performance_baselines", self.conn)
        df.to_csv(output_dir / "performance_baselines.csv", index=False)

        # Export regression alerts
        df = pd.read_sql_query("SELECT * FROM regression_alerts", self.conn)
        df.to_csv(output_dir / "regression_alerts.csv", index=False)

    def close(self):
        """Close database connection"""
        self.conn.close()


def main():
    """Example usage and testing"""
    # This would typically be called by the benchmark runner
    history = BenchmarkHistory()

    # Example: Load and record a benchmark run
    # with open('benchmark_results/standard_latest.json', 'r') as f:
    #     results = json.load(f)
    # run_id = history.record_run(results)

    # Update baselines
    # history.update_baselines('standard')

    # Detect regressions
    # regressions = history.detect_regressions(run_id)
    # for reg in regressions:
    #     print(f"REGRESSION DETECTED: {reg}")

    # Get trend analysis
    # trends = history.get_trend_analysis('standard', 'genetic_algorithm', 'qap_chr12a')
    # print(f"Trends: {trends}")

    # Generate comparison matrix
    # comparison = history.generate_comparison_matrix('standard', 'qap')
    # print(comparison)

    history.close()


if __name__ == "__main__":
    main()