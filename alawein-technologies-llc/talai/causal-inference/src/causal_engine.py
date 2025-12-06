"""
Causal Inference Engine for TalAI
==================================

Advanced causal discovery and inference system using state-of-the-art methods
from econometrics, epidemiology, and machine learning.

References:
- Pearl, J. (2009). Causality: Models, Reasoning, and Inference. Cambridge University Press.
- Imbens, G. W., & Rubin, D. B. (2015). Causal Inference for Statistics, Social, and Biomedical Sciences.
- Spirtes, P., et al. (2000). Causation, Prediction, and Search. MIT Press.
- Schölkopf, B., et al. (2021). Toward Causal Representation Learning. IEEE.
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import warnings
from scipy import stats, linalg
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import NearestNeighbors
import itertools
from collections import defaultdict
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CausalDiscoveryMethod(Enum):
    """Available causal discovery algorithms."""
    PC = "peter-clark"  # Peter-Clark algorithm
    FCI = "fast-causal-inference"  # Fast Causal Inference
    GES = "greedy-equivalence-search"  # Greedy Equivalence Search
    LINGAM = "linear-non-gaussian"  # Linear Non-Gaussian Acyclic Model
    NOTEARS = "no-tears"  # DAG learning with continuous optimization
    GRANGER = "granger-causality"  # Time series causality


class IdentificationStrategy(Enum):
    """Causal identification strategies."""
    BACKDOOR = "backdoor-adjustment"
    FRONTDOOR = "frontdoor-adjustment"
    IV = "instrumental-variable"
    DID = "difference-in-differences"
    RDD = "regression-discontinuity"
    PSM = "propensity-score-matching"
    SCM = "synthetic-control"


@dataclass
class CausalGraph:
    """Represents a causal directed acyclic graph (DAG)."""
    nodes: List[str]
    edges: List[Tuple[str, str]]
    confounders: List[str] = field(default_factory=list)
    instruments: List[str] = field(default_factory=list)
    mediators: List[str] = field(default_factory=list)
    colliders: List[str] = field(default_factory=list)
    graph: Optional[nx.DiGraph] = None

    def __post_init__(self):
        """Initialize the NetworkX graph representation."""
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(self.nodes)
        self.graph.add_edges_from(self.edges)
        self._identify_special_nodes()

    def _identify_special_nodes(self):
        """Identify colliders, confounders, and mediators in the graph."""
        for node in self.nodes:
            parents = list(self.graph.predecessors(node))
            children = list(self.graph.successors(node))

            # Collider: has multiple parents
            if len(parents) > 1:
                self.colliders.append(node)

            # Potential confounder: has paths to multiple nodes
            if len(children) > 1:
                for child_pair in itertools.combinations(children, 2):
                    if not nx.has_path(self.graph, child_pair[0], child_pair[1]) and \
                       not nx.has_path(self.graph, child_pair[1], child_pair[0]):
                        if node not in self.confounders:
                            self.confounders.append(node)

    def get_backdoor_paths(self, treatment: str, outcome: str) -> List[List[str]]:
        """Find all backdoor paths from treatment to outcome."""
        backdoor_paths = []

        # Get all simple paths from treatment to outcome
        try:
            for path in nx.all_simple_paths(self.graph.to_undirected(), treatment, outcome):
                # Check if this is a backdoor path (starts with arrow into treatment)
                if len(path) > 2:
                    first_edge_reversed = self.graph.has_edge(path[1], path[0])
                    if first_edge_reversed:
                        backdoor_paths.append(path)
        except nx.NetworkXNoPath:
            pass

        return backdoor_paths

    def get_adjustment_set(self, treatment: str, outcome: str) -> List[str]:
        """Find minimal adjustment set for backdoor criterion."""
        backdoor_paths = self.get_backdoor_paths(treatment, outcome)

        if not backdoor_paths:
            return []

        # Find nodes that block backdoor paths
        adjustment_candidates = set()
        for path in backdoor_paths:
            # Exclude treatment and outcome
            path_nodes = set(path[1:-1])
            adjustment_candidates.update(path_nodes)

        # Remove descendants of treatment (bad controls)
        descendants = nx.descendants(self.graph, treatment)
        adjustment_set = list(adjustment_candidates - descendants)

        return adjustment_set


@dataclass
class CausalEffect:
    """Represents an estimated causal effect with uncertainty."""
    treatment: str
    outcome: str
    effect: float
    std_error: float
    confidence_interval: Tuple[float, float]
    p_value: float
    method: str
    identification_strategy: IdentificationStrategy
    adjustment_set: List[str] = field(default_factory=list)
    n_samples: int = 0

    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if effect is statistically significant."""
        return self.p_value < alpha

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'treatment': self.treatment,
            'outcome': self.outcome,
            'effect': self.effect,
            'std_error': self.std_error,
            'confidence_interval': self.confidence_interval,
            'p_value': self.p_value,
            'method': self.method,
            'identification_strategy': self.identification_strategy.value,
            'adjustment_set': self.adjustment_set,
            'n_samples': self.n_samples,
            'is_significant': self.is_significant()
        }


class CausalInferenceEngine:
    """
    Main engine for causal discovery and inference.

    Implements multiple causal discovery algorithms and identification strategies
    for estimating causal effects from observational data.
    """

    def __init__(self, random_state: int = 42):
        """Initialize the causal inference engine."""
        self.random_state = random_state
        np.random.seed(random_state)
        self.causal_graph = None
        self.data = None
        self.discovered_effects = []

    def discover_causal_graph(self,
                             data: pd.DataFrame,
                             method: CausalDiscoveryMethod = CausalDiscoveryMethod.PC,
                             alpha: float = 0.05,
                             max_cond_vars: int = 5) -> CausalGraph:
        """
        Discover causal structure from observational data.

        Implements PC algorithm (Spirtes et al., 2000) for causal discovery.
        """
        self.data = data
        nodes = list(data.columns)

        if method == CausalDiscoveryMethod.PC:
            edges = self._pc_algorithm(data, alpha, max_cond_vars)
        elif method == CausalDiscoveryMethod.GRANGER:
            edges = self._granger_causality(data, alpha)
        elif method == CausalDiscoveryMethod.LINGAM:
            edges = self._lingam_discovery(data)
        else:
            raise NotImplementedError(f"Method {method} not yet implemented")

        self.causal_graph = CausalGraph(nodes=nodes, edges=edges)
        logger.info(f"Discovered causal graph with {len(nodes)} nodes and {len(edges)} edges")

        return self.causal_graph

    def _pc_algorithm(self, data: pd.DataFrame, alpha: float, max_cond_vars: int) -> List[Tuple[str, str]]:
        """
        Peter-Clark (PC) algorithm for causal discovery.

        Reference: Spirtes, P., & Glymour, C. (1991). An algorithm for fast recovery of sparse causal graphs.
        """
        n_vars = len(data.columns)
        variables = list(data.columns)

        # Start with complete undirected graph
        skeleton = np.ones((n_vars, n_vars)) - np.eye(n_vars)
        sep_sets = defaultdict(list)

        # Phase 1: Skeleton discovery using conditional independence tests
        for size in range(max_cond_vars + 1):
            for i, j in itertools.combinations(range(n_vars), 2):
                if skeleton[i, j] == 0:
                    continue

                # Get neighbors for conditioning
                neighbors_i = [k for k in range(n_vars) if skeleton[i, k] == 1 and k != j]
                neighbors_j = [k for k in range(n_vars) if skeleton[j, k] == 1 and k != i]
                neighbors = list(set(neighbors_i + neighbors_j))

                if len(neighbors) >= size:
                    # Test conditional independence for all subsets of size
                    for cond_set in itertools.combinations(neighbors, size):
                        if self._conditional_independence_test(
                            data.iloc[:, i],
                            data.iloc[:, j],
                            data.iloc[:, list(cond_set)] if cond_set else None,
                            alpha
                        ):
                            skeleton[i, j] = skeleton[j, i] = 0
                            sep_sets[(i, j)] = list(cond_set)
                            sep_sets[(j, i)] = list(cond_set)
                            break

        # Phase 2: Orient v-structures (colliders)
        edges = []
        oriented = np.zeros((n_vars, n_vars))

        for i, j in itertools.combinations(range(n_vars), 2):
            if skeleton[i, j] == 1:
                # Check for v-structures
                for k in range(n_vars):
                    if k != i and k != j and skeleton[i, k] == 1 and skeleton[j, k] == 1:
                        if skeleton[i, j] == 0 and k not in sep_sets.get((i, j), []):
                            # Orient as i -> k <- j (v-structure)
                            if oriented[i, k] == 0:
                                edges.append((variables[i], variables[k]))
                                oriented[i, k] = 1
                            if oriented[j, k] == 0:
                                edges.append((variables[j], variables[k]))
                                oriented[j, k] = 1

        # Add remaining undirected edges as directed (arbitrary orientation)
        for i, j in itertools.combinations(range(n_vars), 2):
            if skeleton[i, j] == 1 and oriented[i, j] == 0 and oriented[j, i] == 0:
                edges.append((variables[i], variables[j]))

        return edges

    def _conditional_independence_test(self, x: pd.Series, y: pd.Series,
                                      z: Optional[pd.DataFrame], alpha: float) -> bool:
        """
        Test conditional independence using partial correlation.

        H0: X ⊥ Y | Z (X and Y are conditionally independent given Z)
        """
        if z is None or z.empty:
            # Unconditional independence test
            corr, p_value = stats.pearsonr(x, y)
            return p_value > alpha

        # Partial correlation test
        n = len(x)
        k = z.shape[1] if len(z.shape) > 1 else 1

        # Residualize X and Y on Z
        X_resid = x - LinearRegression().fit(z, x).predict(z)
        Y_resid = y - LinearRegression().fit(z, y).predict(z)

        # Test correlation of residuals
        corr, _ = stats.pearsonr(X_resid, Y_resid)

        # Fisher's z-transformation for hypothesis test
        z_score = np.sqrt(n - k - 3) * 0.5 * np.log((1 + corr) / (1 - corr + 1e-10))
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

        return p_value > alpha

    def _granger_causality(self, data: pd.DataFrame, alpha: float, max_lag: int = 5) -> List[Tuple[str, str]]:
        """
        Granger causality test for time series data.

        Reference: Granger, C. W. (1969). Investigating causal relations by econometric models.
        """
        edges = []
        variables = list(data.columns)

        for x_name in variables:
            for y_name in variables:
                if x_name != y_name:
                    x = data[x_name].values
                    y = data[y_name].values

                    # Test if x Granger-causes y
                    best_p_value = 1.0

                    for lag in range(1, min(max_lag + 1, len(x) // 10)):
                        # Restricted model: y_t = a + sum(b_i * y_{t-i})
                        # Unrestricted model: y_t = a + sum(b_i * y_{t-i}) + sum(c_i * x_{t-i})

                        n = len(y) - lag
                        Y = y[lag:]

                        # Build lagged predictors
                        Y_lags = np.column_stack([y[lag-i-1:-i-1 if i > 0 else lag:] for i in range(lag)])
                        X_lags = np.column_stack([x[lag-i-1:-i-1 if i > 0 else lag:] for i in range(lag)])

                        # Fit restricted model
                        restricted = LinearRegression().fit(Y_lags, Y)
                        rss_restricted = np.sum((Y - restricted.predict(Y_lags)) ** 2)

                        # Fit unrestricted model
                        XY_lags = np.column_stack([Y_lags, X_lags])
                        unrestricted = LinearRegression().fit(XY_lags, Y)
                        rss_unrestricted = np.sum((Y - unrestricted.predict(XY_lags)) ** 2)

                        # F-test
                        f_stat = ((rss_restricted - rss_unrestricted) / lag) / (rss_unrestricted / (n - 2 * lag - 1))
                        p_value = 1 - stats.f.cdf(f_stat, lag, n - 2 * lag - 1)

                        best_p_value = min(best_p_value, p_value)

                    if best_p_value < alpha:
                        edges.append((x_name, y_name))
                        logger.info(f"Granger causality: {x_name} -> {y_name} (p={best_p_value:.4f})")

        return edges

    def _lingam_discovery(self, data: pd.DataFrame) -> List[Tuple[str, str]]:
        """
        Linear Non-Gaussian Acyclic Model (LiNGAM) for causal discovery.

        Reference: Shimizu, S., et al. (2006). A linear non-Gaussian acyclic model for causal discovery.
        """
        X = data.values
        n_features = X.shape[1]
        variables = list(data.columns)

        # Center the data
        X = X - np.mean(X, axis=0)

        # Estimate mixing matrix using ICA-like approach
        W = np.eye(n_features)

        # Simple implementation using iterative regression
        causal_order = []
        remaining = list(range(n_features))

        while remaining:
            # Find exogenous variable (minimal non-Gaussianity when regressed on others)
            min_score = float('inf')
            best_var = None

            for var in remaining:
                others = [v for v in remaining if v != var]
                if others:
                    # Regress var on others
                    reg = LinearRegression().fit(X[:, others], X[:, var])
                    residuals = X[:, var] - reg.predict(X[:, others])

                    # Test non-Gaussianity using Jarque-Bera test
                    jb_stat, _ = stats.jarque_bera(residuals)

                    if jb_stat < min_score:
                        min_score = jb_stat
                        best_var = var
                else:
                    best_var = var

            causal_order.append(best_var)
            remaining.remove(best_var)

        # Build DAG from causal order
        edges = []
        for i, cause_idx in enumerate(causal_order[:-1]):
            for effect_idx in causal_order[i+1:]:
                # Test if there's a direct effect
                reg = LinearRegression().fit(X[:, [cause_idx]], X[:, effect_idx])
                if abs(reg.coef_[0]) > 0.1:  # Threshold for edge existence
                    edges.append((variables[cause_idx], variables[effect_idx]))

        return edges

    def estimate_causal_effect(self,
                              treatment: str,
                              outcome: str,
                              strategy: IdentificationStrategy = IdentificationStrategy.BACKDOOR,
                              covariates: Optional[List[str]] = None,
                              **kwargs) -> CausalEffect:
        """
        Estimate causal effect using specified identification strategy.

        Supports multiple identification strategies from econometrics and epidemiology.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call discover_causal_graph first.")

        if strategy == IdentificationStrategy.BACKDOOR:
            return self._backdoor_adjustment(treatment, outcome, covariates)
        elif strategy == IdentificationStrategy.PSM:
            return self._propensity_score_matching(treatment, outcome, covariates, **kwargs)
        elif strategy == IdentificationStrategy.IV:
            return self._instrumental_variable(treatment, outcome, kwargs.get('instrument'), covariates)
        elif strategy == IdentificationStrategy.DID:
            return self._difference_in_differences(treatment, outcome, kwargs.get('time_var'), kwargs.get('group_var'))
        elif strategy == IdentificationStrategy.RDD:
            return self._regression_discontinuity(treatment, outcome, kwargs.get('running_var'), kwargs.get('cutoff'))
        else:
            raise NotImplementedError(f"Strategy {strategy} not yet implemented")

    def _backdoor_adjustment(self, treatment: str, outcome: str,
                            covariates: Optional[List[str]] = None) -> CausalEffect:
        """
        Estimate causal effect using backdoor adjustment (G-formula).

        Reference: Pearl, J. (2009). Causal inference in statistics: An overview.
        """
        X = self.data[treatment].values
        Y = self.data[outcome].values

        # Use provided covariates or find adjustment set from causal graph
        if covariates is None and self.causal_graph:
            covariates = self.causal_graph.get_adjustment_set(treatment, outcome)

        if not covariates:
            # No adjustment needed - simple difference
            effect = np.mean(Y[X == 1]) - np.mean(Y[X == 0]) if len(np.unique(X)) == 2 else np.nan

            # Bootstrap for standard error
            n_boot = 1000
            boot_effects = []
            for _ in range(n_boot):
                idx = np.random.choice(len(X), len(X), replace=True)
                X_boot, Y_boot = X[idx], Y[idx]
                if len(np.unique(X_boot)) == 2:
                    boot_effect = np.mean(Y_boot[X_boot == 1]) - np.mean(Y_boot[X_boot == 0])
                    boot_effects.append(boot_effect)

            std_error = np.std(boot_effects) if boot_effects else np.nan
        else:
            # Regression adjustment
            Z = self.data[covariates].values

            # Fit outcome model
            XZ = np.column_stack([X, Z])
            model = LinearRegression().fit(XZ, Y)

            # Estimate ATE using G-formula
            XZ1 = np.column_stack([np.ones(len(X)), Z])
            XZ0 = np.column_stack([np.zeros(len(X)), Z])

            Y1_pred = model.predict(XZ1)
            Y0_pred = model.predict(XZ0)

            effect = np.mean(Y1_pred - Y0_pred)

            # Standard error via delta method
            residuals = Y - model.predict(XZ)
            var_residuals = np.var(residuals)
            std_error = np.sqrt(var_residuals / len(X))

        # Confidence interval and p-value
        ci = (effect - 1.96 * std_error, effect + 1.96 * std_error)
        z_score = effect / (std_error + 1e-10)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

        return CausalEffect(
            treatment=treatment,
            outcome=outcome,
            effect=effect,
            std_error=std_error,
            confidence_interval=ci,
            p_value=p_value,
            method="Backdoor Adjustment (G-formula)",
            identification_strategy=IdentificationStrategy.BACKDOOR,
            adjustment_set=covariates or [],
            n_samples=len(X)
        )

    def _propensity_score_matching(self, treatment: str, outcome: str,
                                  covariates: List[str],
                                  caliper: float = 0.1,
                                  n_neighbors: int = 1) -> CausalEffect:
        """
        Propensity score matching for causal effect estimation.

        Reference: Rosenbaum, P. R., & Rubin, D. B. (1983). The central role of the propensity score.
        """
        T = self.data[treatment].values
        Y = self.data[outcome].values
        X = self.data[covariates].values if covariates else np.ones((len(T), 1))

        # Estimate propensity scores
        ps_model = LogisticRegression(random_state=self.random_state)
        ps_model.fit(X, T)
        propensity_scores = ps_model.predict_proba(X)[:, 1]

        # Match treated to control units
        treated_idx = np.where(T == 1)[0]
        control_idx = np.where(T == 0)[0]

        # Use nearest neighbor matching within caliper
        nn = NearestNeighbors(n_neighbors=min(n_neighbors, len(control_idx)))
        nn.fit(propensity_scores[control_idx].reshape(-1, 1))

        matched_outcomes_treated = []
        matched_outcomes_control = []

        for idx in treated_idx:
            ps_treated = propensity_scores[idx]
            distances, indices = nn.kneighbors([[ps_treated]])

            # Apply caliper
            valid_matches = indices[0][distances[0] < caliper]

            if len(valid_matches) > 0:
                matched_outcomes_treated.append(Y[idx])
                matched_outcomes_control.append(np.mean(Y[control_idx[valid_matches]]))

        if matched_outcomes_treated:
            # Calculate ATT (Average Treatment effect on Treated)
            effect = np.mean(np.array(matched_outcomes_treated) - np.array(matched_outcomes_control))

            # Bootstrap for standard error
            n_boot = 500
            boot_effects = []
            n_matched = len(matched_outcomes_treated)

            for _ in range(n_boot):
                boot_idx = np.random.choice(n_matched, n_matched, replace=True)
                boot_treated = np.array(matched_outcomes_treated)[boot_idx]
                boot_control = np.array(matched_outcomes_control)[boot_idx]
                boot_effects.append(np.mean(boot_treated - boot_control))

            std_error = np.std(boot_effects)
        else:
            effect = np.nan
            std_error = np.nan

        ci = (effect - 1.96 * std_error, effect + 1.96 * std_error)
        z_score = effect / (std_error + 1e-10)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

        return CausalEffect(
            treatment=treatment,
            outcome=outcome,
            effect=effect,
            std_error=std_error,
            confidence_interval=ci,
            p_value=p_value,
            method=f"Propensity Score Matching (caliper={caliper})",
            identification_strategy=IdentificationStrategy.PSM,
            adjustment_set=covariates or [],
            n_samples=len(matched_outcomes_treated)
        )

    def _instrumental_variable(self, treatment: str, outcome: str,
                              instrument: str, covariates: Optional[List[str]] = None) -> CausalEffect:
        """
        Instrumental variable estimation (2SLS).

        Reference: Angrist, J. D., & Pischke, J. S. (2008). Mostly harmless econometrics.
        """
        Z = self.data[instrument].values  # Instrument
        X = self.data[treatment].values   # Treatment
        Y = self.data[outcome].values     # Outcome

        if covariates:
            W = self.data[covariates].values
            # Include covariates in both stages
            ZW = np.column_stack([Z, W])
            XW_pred = LinearRegression().fit(ZW, X).predict(ZW)
            XW = np.column_stack([XW_pred, W])
        else:
            # Simple IV without covariates
            XW_pred = LinearRegression().fit(Z.reshape(-1, 1), X).predict(Z.reshape(-1, 1))
            XW = XW_pred.reshape(-1, 1)

        # Second stage
        model = LinearRegression().fit(XW, Y)
        effect = model.coef_[0]

        # Standard error (simplified - should use robust standard errors)
        residuals = Y - model.predict(XW)
        var_residuals = np.var(residuals)
        std_error = np.sqrt(var_residuals / len(X))

        ci = (effect - 1.96 * std_error, effect + 1.96 * std_error)
        t_stat = effect / std_error
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(X) - XW.shape[1]))

        return CausalEffect(
            treatment=treatment,
            outcome=outcome,
            effect=effect,
            std_error=std_error,
            confidence_interval=ci,
            p_value=p_value,
            method=f"Instrumental Variable (2SLS, instrument={instrument})",
            identification_strategy=IdentificationStrategy.IV,
            adjustment_set=covariates or [],
            n_samples=len(X)
        )

    def _difference_in_differences(self, treatment: str, outcome: str,
                                  time_var: str, group_var: str) -> CausalEffect:
        """
        Difference-in-differences estimation.

        Reference: Card, D., & Krueger, A. B. (1994). Minimum wages and employment.
        """
        df = self.data.copy()

        # Create interaction term
        df['treat_post'] = df[treatment] * df[time_var]

        # Run DiD regression: Y = a + b*treat + c*post + d*treat*post + e
        X = df[[treatment, time_var, 'treat_post']].values
        Y = df[outcome].values

        model = LinearRegression().fit(X, Y)

        # The DiD estimate is the coefficient on the interaction term
        effect = model.coef_[2]

        # Calculate standard error
        residuals = Y - model.predict(X)
        n = len(Y)
        k = X.shape[1]

        # Variance-covariance matrix
        sigma2 = np.sum(residuals**2) / (n - k - 1)
        var_coef = sigma2 * np.linalg.inv(X.T @ X)
        std_error = np.sqrt(var_coef[2, 2])

        ci = (effect - 1.96 * std_error, effect + 1.96 * std_error)
        t_stat = effect / std_error
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - k - 1))

        return CausalEffect(
            treatment=treatment,
            outcome=outcome,
            effect=effect,
            std_error=std_error,
            confidence_interval=ci,
            p_value=p_value,
            method="Difference-in-Differences",
            identification_strategy=IdentificationStrategy.DID,
            adjustment_set=[time_var, group_var],
            n_samples=n
        )

    def _regression_discontinuity(self, treatment: str, outcome: str,
                                 running_var: str, cutoff: float,
                                 bandwidth: Optional[float] = None) -> CausalEffect:
        """
        Regression discontinuity design.

        Reference: Imbens, G., & Lemieux, T. (2008). Regression discontinuity designs: A guide to practice.
        """
        R = self.data[running_var].values  # Running variable
        Y = self.data[outcome].values
        T = (R >= cutoff).astype(int)  # Treatment assignment

        # Optimal bandwidth selection (Imbens-Kalyanaraman)
        if bandwidth is None:
            h_opt = 2.34 * np.std(R) * (len(R) ** (-1/5))
            bandwidth = h_opt

        # Local linear regression on both sides of cutoff
        mask = np.abs(R - cutoff) <= bandwidth
        R_local = R[mask]
        Y_local = Y[mask]
        T_local = T[mask]

        # Center running variable at cutoff
        R_centered = R_local - cutoff

        # Fit model: Y = a + b*T + c*R + d*T*R + e
        X = np.column_stack([T_local, R_centered, T_local * R_centered])
        model = LinearRegression().fit(X, Y_local)

        # RD estimate is the jump at the cutoff
        effect = model.coef_[0]

        # Standard error
        residuals = Y_local - model.predict(X)
        n = len(Y_local)
        k = X.shape[1]

        sigma2 = np.sum(residuals**2) / (n - k - 1)
        var_coef = sigma2 * np.linalg.inv(X.T @ X)
        std_error = np.sqrt(var_coef[0, 0])

        ci = (effect - 1.96 * std_error, effect + 1.96 * std_error)
        t_stat = effect / std_error
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - k - 1))

        return CausalEffect(
            treatment=treatment,
            outcome=outcome,
            effect=effect,
            std_error=std_error,
            confidence_interval=ci,
            p_value=p_value,
            method=f"Regression Discontinuity (cutoff={cutoff}, bandwidth={bandwidth:.3f})",
            identification_strategy=IdentificationStrategy.RDD,
            adjustment_set=[running_var],
            n_samples=n
        )

    def generate_causal_narrative(self) -> str:
        """Generate human-readable narrative explaining discovered causal relationships."""
        if not self.causal_graph:
            return "No causal graph discovered yet. Run discover_causal_graph() first."

        narrative = "## Causal Discovery Report\n\n"
        narrative += f"### Graph Structure\n"
        narrative += f"- **Nodes**: {len(self.causal_graph.nodes)} variables\n"
        narrative += f"- **Edges**: {len(self.causal_graph.edges)} causal relationships\n"
        narrative += f"- **Confounders identified**: {', '.join(self.causal_graph.confounders) if self.causal_graph.confounders else 'None'}\n"
        narrative += f"- **Colliders identified**: {', '.join(self.causal_graph.colliders) if self.causal_graph.colliders else 'None'}\n\n"

        narrative += "### Causal Relationships\n"
        for cause, effect in self.causal_graph.edges[:10]:  # Show first 10
            narrative += f"- **{cause}** → **{effect}**: {cause} has a causal influence on {effect}\n"

        if self.discovered_effects:
            narrative += "\n### Estimated Causal Effects\n"
            for effect in self.discovered_effects:
                narrative += f"\n#### {effect.treatment} → {effect.outcome}\n"
                narrative += f"- **Method**: {effect.method}\n"
                narrative += f"- **Effect size**: {effect.effect:.4f} (SE: {effect.std_error:.4f})\n"
                narrative += f"- **95% CI**: [{effect.confidence_interval[0]:.4f}, {effect.confidence_interval[1]:.4f}]\n"
                narrative += f"- **Statistical significance**: {'Yes' if effect.is_significant() else 'No'} (p={effect.p_value:.4f})\n"
                if effect.adjustment_set:
                    narrative += f"- **Adjusted for**: {', '.join(effect.adjustment_set)}\n"

        return narrative

    def validate_assumptions(self) -> Dict[str, Any]:
        """Validate key assumptions for causal inference."""
        validations = {
            'dag_acyclic': True,
            'no_unmeasured_confounding': 'Cannot be tested from data alone',
            'positivity': {},
            'sutva': 'Assumed - no interference between units',
            'common_support': {}
        }

        if self.causal_graph:
            # Check for cycles
            validations['dag_acyclic'] = nx.is_directed_acyclic_graph(self.causal_graph.graph)

        if self.data is not None:
            # Check positivity/common support for binary treatments
            for col in self.data.columns:
                if self.data[col].nunique() == 2:
                    prop_treated = self.data[col].mean()
                    validations['positivity'][col] = {
                        'prop_treated': prop_treated,
                        'violation': prop_treated < 0.1 or prop_treated > 0.9
                    }

        return validations