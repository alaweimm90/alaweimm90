"""
ASlib Scenario Parser

Parses ASlib ARFF files and provides data in format compatible
with Librex.Meta evaluation pipeline.

ASlib format reference: https://github.com/mlindauer/aslib_data
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import re


class ARFFParser:
    """Robust ARFF file parser for ASlib scenarios.

    Handles:
    - Case-insensitive directives
    - Quoted attribute names and values (single or double quotes)
    - Missing values marked as '?' (converted to NaN)
    - Inline commas inside quoted values
    - Basic type inference for numeric columns
    """

    _ATTR_REGEX = re.compile(
        r"^\s*@attribute\s+(?:'([^']+)'|\"([^\"]+)\"|([^\s]+))\s+(.*)$",
        re.IGNORECASE,
    )

    @staticmethod
    def _strip_quotes(val: str) -> str:
        if isinstance(val, str) and len(val) >= 2 and (
            (val[0] == val[-1] == '"') or (val[0] == val[-1] == "'")
        ):
            return val[1:-1]
        return val

    @staticmethod
    def parse_arff(filepath: str) -> pd.DataFrame:
        """
        Parse an ARFF file into a pandas DataFrame

        Args:
            filepath: Path to .arff file

        Returns:
            df: DataFrame with parsed data
        """
        try:
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
        except FileNotFoundError as e:
            raise FileNotFoundError(f"ARFF file not found: {filepath}") from e

        # Locate @DATA section (case-insensitive)
        data_idx = None
        for i, raw in enumerate(lines):
            line = raw.strip()
            if line.startswith('%') or not line:
                continue
            if line.upper().startswith('@DATA'):
                data_idx = i + 1
                break

        if data_idx is None:
            raise ValueError(f"No @DATA section found in {filepath}")

        # Parse attributes (supports quoted names)
        attributes: List[str] = []
        for raw in lines[:data_idx]:
            m = ARFFParser._ATTR_REGEX.match(raw)
            if not m:
                continue
            name = next((g for g in m.groups()[:3] if g is not None), None)
            if name is None:
                continue
            attributes.append(name)

        if not attributes:
            raise ValueError(f"No @ATTRIBUTE declarations found in {filepath}")

        # Stream-parse the @DATA rows using csv to respect quotes/commas
        import csv

        data_rows: List[List[str]] = []
        reader = csv.reader(
            (l for l in lines[data_idx:] if l.strip() and not l.lstrip().startswith('%')),
            delimiter=',',
            quotechar='"',
            skipinitialspace=True,
        )
        for row in reader:
            if not row:
                continue
            # Strip surrounding quotes from each cell
            row = [ARFFParser._strip_quotes(v).strip() for v in row]
            # Normalize missing values
            row = [None if v in ('?', '') else v for v in row]

            # Align row length to attributes: pad/truncate as needed
            if len(row) < len(attributes):
                row.extend([None] * (len(attributes) - len(row)))
            elif len(row) > len(attributes):
                row = row[: len(attributes)]

            data_rows.append(row)

        # Create DataFrame
        try:
            df = pd.DataFrame(data_rows, columns=attributes)
        except Exception as e:
            raise ValueError(
                f"Failed to construct DataFrame from {filepath}: {e}"
            ) from e

        # Known string/categorical columns in ASlib
        string_columns = {'instance_id', 'algorithm', 'runstatus', 'type', 'repetition'}

        # Convert likely-numeric columns with robust inference
        for col in list(df.columns):
            if col in string_columns:
                continue
            # Try numeric conversion
            converted = pd.to_numeric(df[col], errors='coerce')
            valid_ratio = converted.notna().mean() if len(converted) else 0.0
            if valid_ratio >= 0.5:
                # Use float32 to reduce memory footprint
                df[col] = converted.astype('float32')
            else:
                # Keep as string/object; strip any quotes
                df[col] = df[col].astype('string')

        # Normalize common categorical columns
        for cat_col in ('algorithm', 'runstatus', 'type'):
            if cat_col in df.columns:
                df[cat_col] = df[cat_col].astype('string')

        return df


class ASLibScenarioLoader:
    """
    Loads and processes ASlib algorithm selection scenarios

    Handles:
    - Feature extraction from feature_values.arff
    - Performance data from algorithm_runs.arff
    - Cross-validation splits from cv.arff
    - Timeout handling and Par10 computation
    """

    def __init__(self, scenario_path: str):
        """
        Initialize loader for a specific scenario

        Args:
            scenario_path: Path to ASlib scenario directory
        """
        self.scenario_path = Path(scenario_path)
        self.scenario_name = self.scenario_path.name

        # Load metadata
        self.metadata = self._load_description()

        # Load data
        self.features_df = None
        self.performance_df = None
        self.cv_splits = None

        self._load_data()

    def _load_description(self) -> Dict:
        """Load scenario description (simplified YAML parser)"""
        desc_file = self.scenario_path / "description.txt"
        metadata = {}

        if desc_file.exists():
            with open(desc_file, 'r') as f:
                for line in f:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        metadata[key.strip()] = value.strip()

        return metadata

    def _load_data(self):
        """Load all scenario data files"""
        # Load features
        features_file = self.scenario_path / "feature_values.arff"
        if features_file.exists():
            self.features_df = ARFFParser.parse_arff(str(features_file))
            # Speed up lookups by indexing on instance_id if present
            if 'instance_id' in self.features_df.columns:
                try:
                    self.features_df.set_index('instance_id', inplace=True, drop=False)
                except Exception:
                    pass
            # Cache feature columns (numeric) for vectorized extraction
            exclude_cols = {'instance_id', 'repetition'}
            self._feature_cols = [
                c for c in self.features_df.columns
                if c not in exclude_cols and pd.api.types.is_numeric_dtype(self.features_df[c])
            ]

        # Load performance data
        perf_file = self.scenario_path / "algorithm_runs.arff"
        if perf_file.exists():
            self.performance_df = ARFFParser.parse_arff(str(perf_file))
            # Category dtype for algorithm to reduce memory
            if 'algorithm' in self.performance_df.columns:
                try:
                    self.performance_df['algorithm'] = self.performance_df['algorithm'].astype('category')
                except Exception:
                    pass
            # Index by instance for faster lookup
            if 'instance_id' in self.performance_df.columns:
                try:
                    self.performance_df.set_index('instance_id', inplace=True, drop=False)
                except Exception:
                    pass
            # Cache inferred metric column
            self._metric_col = self._infer_metric_col()
            self._has_runstatus = 'runstatus' in self.performance_df.columns

        # Load CV splits
        cv_file = self.scenario_path / "cv.arff"
        if cv_file.exists():
            self.cv_splits = ARFFParser.parse_arff(str(cv_file))
            # Normalize fold/type casing
            if 'type' in self.cv_splits.columns:
                self.cv_splits['type'] = self.cv_splits['type'].str.lower()

    def get_instance_ids(self) -> List[str]:
        """Get list of all instance IDs"""
        if self.features_df is not None and 'instance_id' in self.features_df.columns:
            return pd.Index(self.features_df['instance_id']).unique().tolist()
        return []

    def _infer_metric_col(self) -> Optional[str]:
        if self.performance_df is None:
            return None
        for cand in ('runtime', 'runlength', 'cost', 'quality'):
            if cand in self.performance_df.columns and pd.api.types.is_numeric_dtype(self.performance_df[cand]):
                return cand
        for col in self.performance_df.columns:
            if col in ('instance_id', 'algorithm', 'runstatus', 'repetition'):
                continue
            if pd.api.types.is_numeric_dtype(self.performance_df[col]):
                return col
        return None

    def get_algorithm_names(self) -> List[str]:
        """Get list of all algorithm names"""
        if self.performance_df is not None:
            return self.performance_df['algorithm'].unique().tolist()
        return []

    def get_features(self, instance_id: str) -> np.ndarray:
        """
        Get feature vector for an instance

        Args:
            instance_id: Instance identifier

        Returns:
            features: Numpy array of features
        """
        if self.features_df is None:
            return np.array([])

        # Fast path: use index if available
        try:
            if self.features_df.index.name == 'instance_id':
                instance_row = self.features_df.loc[[instance_id]]
            else:
                instance_row = self.features_df[
                    self.features_df['instance_id'] == instance_id
                ]
        except Exception:
            instance_row = self.features_df[
                self.features_df['instance_id'] == instance_id
            ]

        if len(instance_row) == 0:
            return np.array([])

        # Get numeric columns (excluding instance_id, repetition)
        exclude_cols = ['instance_id', 'repetition']
        feature_cols = [
            col for col in self.features_df.columns
            if col not in exclude_cols and pd.api.types.is_numeric_dtype(self.features_df[col])
        ]

        if not feature_cols:
            return np.array([])

        features = instance_row[feature_cols].values[0]

        # Handle NaN values
        features = np.nan_to_num(features, nan=0.0)

        return features.astype(float)

    def get_performance(
        self,
        instance_id: str,
        algorithm: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Get performance data for an instance

        Args:
            instance_id: Instance identifier
            algorithm: Specific algorithm (None = all algorithms)

        Returns:
            performance: Dict of {algorithm_name: runtime}
        """
        if self.performance_df is None:
            return {}

        # Filter by instance (use index if available)
        try:
            if self.performance_df.index.name == 'instance_id':
                perf_data = self.performance_df.loc[[instance_id]]
            else:
                perf_data = self.performance_df[self.performance_df['instance_id'] == instance_id]
        except Exception:
            perf_data = self.performance_df[self.performance_df['instance_id'] == instance_id]

        if algorithm is not None:
            perf_data = perf_data[perf_data['algorithm'] == algorithm]

        # Decide metric column (prefer 'runtime', then 'runlength', 'cost', or first numeric)
        metric_col = getattr(self, '_metric_col', None) or self._infer_metric_col()
        if metric_col is None:
            return {}

        # Extract metric per algorithm with PAR10 handling
        performance: Dict[str, float] = {}
        timeout = self.get_timeout()
        for _, row in perf_data.iterrows():
            algo_name = str(row['algorithm'])
            status = str(row['runstatus']).lower() if self._has_runstatus and row['runstatus'] is not None else 'ok'
            try:
                value = float(row[metric_col])
            except Exception:
                value = float('nan')

            if status in {'timeout', 'memout', 'crash'} or pd.isna(value):
                value = 10.0 * float(timeout)

            performance[algo_name] = float(value)

        return performance

    def get_timeout(self) -> float:
        """Get timeout value from metadata"""
        timeout_str = self.metadata.get('algorithm_cutoff_time', '5000')
        try:
            return float(timeout_str)
        except (ValueError, TypeError):
            return 5000.0  # Default

    def get_best_algorithm(self, instance_id: str) -> str:
        """
        Get best performing algorithm for an instance

        Args:
            instance_id: Instance identifier

        Returns:
            best_algo: Name of best algorithm (minimum runtime)
        """
        performance = self.get_performance(instance_id)
        if not performance:
            return None

        return min(performance, key=performance.get)

    def get_training_data(
        self,
        instance_ids: Optional[List[str]] = None,
        max_instances: Optional[int] = None
    ) -> List[Dict]:
        """
        Get training data in Librex.Meta format

        Args:
            instance_ids: Specific instances to include (None = all)
            max_instances: Maximum number of instances to return

        Returns:
            training_data: List of dicts with 'instance', 'features', 'performances'
        """
        if instance_ids is None:
            instance_ids = self.get_instance_ids()

        if max_instances is not None:
            instance_ids = instance_ids[:max_instances]

        training_data = []

        for inst_id in instance_ids:
            features = self.get_features(inst_id)
            performance = self.get_performance(inst_id)

            if len(features) == 0 or len(performance) == 0:
                continue

            # Convert runtime to normalized performance score (lower is better)
            # Score = 1 / (1 + runtime) so faster solvers have higher scores
            performances_normalized = {}
            for algo, runtime in performance.items():
                score = 1.0 / (1.0 + runtime)
                performances_normalized[algo] = score

            training_data.append({
                'instance_id': inst_id,
                'features': features,
                'performances': performances_normalized,
                'raw_runtimes': performance
            })

        return training_data

    def get_training_arrays(
        self,
        instance_ids: Optional[List[str]] = None,
        max_instances: Optional[int] = None
    ) -> Tuple[List[str], np.ndarray, List[str], np.ndarray, np.ndarray]:
        """
        Vectorized training export for performance-critical pipelines.

        Returns:
            ids: ordered list of instance_ids
            X: features matrix [n_instances, n_features]
            algorithms: ordered list of algorithm names (columns in Y and R)
            Y: normalized performance scores [n_instances, n_algorithms]
            R: raw runtimes/costs with PAR10 applied [n_instances, n_algorithms]
        """
        ids = instance_ids or self.get_instance_ids()
        if max_instances is not None:
            ids = ids[:max_instances]

        if not ids:
            return [], np.empty((0, 0), dtype=np.float32), [], np.empty((0, 0), dtype=np.float32), np.empty((0, 0), dtype=np.float32)

        # Features matrix via reindex for ordered selection
        feat_cols = getattr(self, '_feature_cols', None)
        if not feat_cols:
            # Fallback to per-instance path
            X = np.array([self.get_features(i) for i in ids], dtype=float)
        else:
            try:
                df_sel = self.features_df.set_index('instance_id', drop=False).reindex(ids)
            except Exception:
                df_sel = self.features_df[self.features_df['instance_id'].isin(ids)]
                df_sel = df_sel.sort_values('instance_id')
            X = df_sel[feat_cols].astype('float32').to_numpy()
            X = np.nan_to_num(X, nan=0.0).astype('float32')

        # Algorithms order
        algos = self.get_algorithm_names()
        if not algos:
            return ids, X, [], np.empty((len(ids), 0), dtype=np.float32), np.empty((len(ids), 0), dtype=np.float32)

        # Build runtime table [instances x algorithms]
        metric_col = getattr(self, '_metric_col', None) or self._infer_metric_col()
        timeout = float(self.get_timeout())

        # Work on a copy with a simple RangeIndex to avoid index/column ambiguity
        df_runs = self.performance_df.reset_index(drop=True).copy()
        # Compute value with PAR10 handling
        if metric_col not in df_runs.columns:
            # Ensure column exists to avoid KeyErrors
            df_runs[metric_col] = np.nan
        val = pd.to_numeric(df_runs[metric_col], errors='coerce')
        if 'runstatus' in df_runs.columns:
            bad = df_runs['runstatus'].astype(str).str.lower().isin(['timeout', 'memout', 'crash']) | val.isna()
        else:
            bad = val.isna()
        val = val.mask(bad, other=10.0 * timeout)
        df_runs['_value'] = val.astype('float32')

        # Pivot to wide format, align to requested ids and algo order
        wide = df_runs.pivot_table(index='instance_id', columns='algorithm', values='_value', aggfunc='min')
        wide = wide.reindex(index=ids, columns=algos)
        # Fill any remaining NaNs with PAR10
        wide = wide.fillna(10.0 * timeout)
        R = wide.to_numpy(dtype='float32')

        # Normalize
        Y = (1.0 / (1.0 + R)).astype('float32')

        return ids, X, algos, Y, R

    def get_cv_split(self, fold: int = 0) -> Tuple[List[str], List[str]]:
        """
        Get train/test split for cross-validation

        Args:
            fold: Fold number (0-indexed)

        Returns:
            train_ids, test_ids: Lists of instance IDs
        """
        if self.cv_splits is None:
            # Default: 80/20 split
            all_ids = self.get_instance_ids()
            n_train = int(0.8 * len(all_ids))
            return all_ids[:n_train], all_ids[n_train:]

        # Parse cv.arff format (columns: instance_id, repetition, fold, type)
        # Handle fold possibly stored as string
        fold_col = self.cv_splits.get('fold') if isinstance(self.cv_splits, pd.DataFrame) else None
        if fold_col is None:
            all_ids = self.get_instance_ids()
            n_train = int(0.8 * len(all_ids))
            return all_ids[:n_train], all_ids[n_train:]

        # Coerce fold to numeric if needed
        try:
            fold_values = pd.to_numeric(self.cv_splits['fold'], errors='coerce')
            fold_data = self.cv_splits[fold_values == fold]
        except Exception:
            fold_data = self.cv_splits[self.cv_splits['fold'] == fold]

        # Use provided type split if available, else fall back to deterministic 80/20
        if 'type' in fold_data.columns and not fold_data.empty:
            train_ids = fold_data[fold_data['type'] == 'train']['instance_id'].astype(str).tolist()
            test_ids = fold_data[fold_data['type'] == 'test']['instance_id'].astype(str).tolist()
        else:
            ids = (fold_data['instance_id'].astype(str).unique().tolist()
                   if not fold_data.empty else self.get_instance_ids())
            n_train = int(0.8 * len(ids))
            train_ids, test_ids = ids[:n_train], ids[n_train:]

        return train_ids, test_ids

    def get_summary(self) -> Dict:
        """Get scenario summary statistics"""
        return {
            'name': self.scenario_name,
            'n_instances': len(self.get_instance_ids()),
            'n_algorithms': len(self.get_algorithm_names()),
            'n_features': len(self.get_features(self.get_instance_ids()[0])) if self.get_instance_ids() else 0,
            'timeout': self.get_timeout(),
            'algorithms': self.get_algorithm_names()
        }


def main():
    """Test ASlib parser"""
    print("="*70)
    print("ASlib Scenario Parser Test")
    print("="*70)

    # Load SAT11-HAND scenario
    scenario = ASLibScenarioLoader("aslib_data/SAT11-HAND")

    # Print summary
    summary = scenario.get_summary()
    print(f"\nScenario: {summary['name']}")
    print(f"Instances: {summary['n_instances']}")
    print(f"Algorithms: {summary['n_algorithms']}")
    print(f"Features: {summary['n_features']}")
    print(f"Timeout: {summary['timeout']}")

    print(f"\nAlgorithms:")
    for algo in summary['algorithms'][:10]:
        print(f"  - {algo}")
    if len(summary['algorithms']) > 10:
        print(f"  ... and {len(summary['algorithms']) - 10} more")

    # Get training data
    print(f"\nLoading training data...")
    training_data = scenario.get_training_data(max_instances=10)

    print(f"Loaded {len(training_data)} instances")

    if training_data:
        sample = training_data[0]
        print(f"\nSample instance:")
        print(f"  ID: {sample['instance_id']}")
        print(f"  Features shape: {sample['features'].shape}")
        print(f"  Algorithms: {len(sample['performances'])}")
        print(f"  Performance scores: {list(sample['performances'].values())[:5]}")

    print("\n" + "="*70)
    print("✓ ASlib parser working!")
    print("="*70)


if __name__ == "__main__":
    main()
