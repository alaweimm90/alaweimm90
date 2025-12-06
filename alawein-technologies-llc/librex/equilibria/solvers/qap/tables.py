"""
Librex.QAP Tables Module
========================

Publication-quality tables for benchmark results.

Key tables:
- Summary statistics (mean, std, min, max per method)
- Method comparison (novel vs baseline)
- Parameter sweep results
- Ranking tables
- Enhanced pandas formatting

Author: Meshal Alawein
Date: 2025-10-16
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def create_summary_table(
    results_df: pd.DataFrame, group_by: str = "method", metrics: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Create summary statistics table.

    Computes mean, std, min, max for each metric grouped by a column.

    Parameters
    ----------
    results_df : pd.DataFrame
        Benchmark results
    group_by : str
        Column to group by (usually 'method' or 'instance')
    metrics : list of str, optional
        Metrics to summarize
        If None, uses ['objective', 'gap', 'time', 'ds_violation']

    Returns
    -------
    summary : pd.DataFrame
        Summary table with MultiIndex columns (metric, statistic)

    Example
    -------
    >>> summary = create_summary_table(results, group_by='method')
    >>> print(summary)
                    objective               gap                 time
                    mean    std    min     mean    std    min    mean   std
    method
    1-FFT-Laplace   1850.2  45.3  1802   3.2%    1.5%   1.8%   10.2   0.8
    2-Reverse-Time  1872.5  52.1  1815   4.1%    1.8%   2.3%   10.8   1.1
    ...
    """
    if metrics is None:
        metrics = ["objective", "gap", "time", "ds_violation"]

    # Filter to available metrics
    metrics = [m for m in metrics if m in results_df.columns]

    # Group and aggregate
    summary_dict = {}

    for metric in metrics:
        grouped = results_df.groupby(group_by)[metric]
        summary_dict[f"{metric}_mean"] = grouped.mean()
        summary_dict[f"{metric}_std"] = grouped.std()
        summary_dict[f"{metric}_min"] = grouped.min()
        summary_dict[f"{metric}_max"] = grouped.max()

    summary = pd.DataFrame(summary_dict)

    # Reorder columns to group by metric
    ordered_cols = []
    for metric in metrics:
        for stat in ["mean", "std", "min", "max"]:
            col = f"{metric}_{stat}"
            if col in summary.columns:
                ordered_cols.append(col)

    summary = summary[ordered_cols]

    return summary


def create_method_comparison(
    results_df: pd.DataFrame,
    novel_methods: List[str],
    baseline_methods: List[str],
    metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compare novel vs baseline methods.

    Computes aggregate statistics for each group.

    Parameters
    ----------
    results_df : pd.DataFrame
        Benchmark results
    novel_methods : list of str
        List of novel method names
    baseline_methods : list of str
        List of baseline method names
    metrics : list of str, optional
        Metrics to compare

    Returns
    -------
    comparison : pd.DataFrame
        Comparison table with rows ['Novel', 'Baseline', 'Improvement']

    Example
    -------
    >>> comparison = create_method_comparison(results, novel_list, baseline_list)
    >>> print(comparison)
                    objective_mean  gap_mean  time_mean  ds_violation_mean
    Novel           1865.3          3.8%      10.5       0.0032
    Baseline        2150.7          18.2%     11.2       0.0041
    Improvement     -285.4 (-13.3%) -14.4pp   -0.7s      -0.0009
    """
    if metrics is None:
        metrics = ["objective", "gap", "time", "ds_violation"]

    metrics = [m for m in metrics if m in results_df.columns]

    # Separate novel and baseline
    novel_df = results_df[results_df["method"].isin(novel_methods)]
    baseline_df = results_df[results_df["method"].isin(baseline_methods)]

    comparison_data = {}

    for metric in metrics:
        novel_mean = novel_df[metric].mean()
        baseline_mean = baseline_df[metric].mean()
        improvement = novel_mean - baseline_mean
        improvement_pct = 100 * improvement / abs(baseline_mean) if baseline_mean != 0 else 0

        comparison_data[f"{metric}_mean"] = {
            "Novel": novel_mean,
            "Baseline": baseline_mean,
            "Improvement": improvement,
            "Improvement_pct": improvement_pct,
        }

    comparison = pd.DataFrame(comparison_data).T

    return comparison


def create_parameter_sweep_table(
    results_df: pd.DataFrame, param_col: str, metric: str = "gap", group_by: Optional[str] = None
) -> pd.DataFrame:
    """
    Create table for parameter sweep results.

    Shows how metric varies with parameter values.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results with parameter column
    param_col : str
        Name of parameter column (e.g., 'dt', 'mu_entropy')
    metric : str
        Metric to track (e.g., 'gap', 'time')
    group_by : str, optional
        Additional grouping (e.g., 'instance')

    Returns
    -------
    sweep_table : pd.DataFrame
        Parameter sweep results

    Example
    -------
    >>> sweep = create_parameter_sweep_table(results, param_col='dt', metric='gap')
    >>> print(sweep)
        dt      gap_mean  gap_std  gap_min  gap_max
    0   0.001   5.2       1.3      3.1      7.8
    1   0.01    3.8       0.9      2.5      5.1
    2   0.1     4.5       1.5      2.8      6.9
    """
    if group_by:
        grouped = results_df.groupby([param_col, group_by])[metric]
    else:
        grouped = results_df.groupby(param_col)[metric]

    sweep_table = pd.DataFrame(
        {
            f"{metric}_mean": grouped.mean(),
            f"{metric}_std": grouped.std(),
            f"{metric}_min": grouped.min(),
            f"{metric}_max": grouped.max(),
        }
    ).reset_index()

    return sweep_table


def create_ranking_table(
    results_df: pd.DataFrame,
    metric: str = "gap",
    ascending: bool = True,
    method_type_map: Optional[Dict[str, str]] = None,
    top_n: Optional[int] = None,
) -> pd.DataFrame:
    """
    Create ranking table by metric.

    Ranks methods and optionally labels as novel/baseline.

    Parameters
    ----------
    results_df : pd.DataFrame
        Benchmark results
    metric : str
        Metric to rank by (e.g., 'gap', 'time')
    ascending : bool
        Sort order (True = lower is better, False = higher is better)
    method_type_map : dict, optional
        Maps method name to type ('novel' or 'baseline')
    top_n : int, optional
        Only show top N methods

    Returns
    -------
    ranking : pd.DataFrame
        Ranked methods with columns:
        - Rank
        - Method
        - Type (if method_type_map provided)
        - [metric]_mean
        - [metric]_std

    Example
    -------
    >>> ranking = create_ranking_table(results, metric='gap')
    >>> print(ranking)
        Rank  Method              Type      gap_mean  gap_std
    0   1     1-FFT-Laplace      Novel     3.2       1.5
    1   2     2-Reverse-Time     Novel     4.1       1.8
    2   3     B02-Sinkhorn       Baseline  5.7       2.1
    ...
    """
    # Aggregate by method
    method_stats = results_df.groupby("method")[metric].agg(["mean", "std"]).reset_index()
    method_stats.columns = ["method", f"{metric}_mean", f"{metric}_std"]

    # Sort by metric
    method_stats = method_stats.sort_values(f"{metric}_mean", ascending=ascending)

    # Add rank
    method_stats["rank"] = range(1, len(method_stats) + 1)

    # Add type if mapping provided
    if method_type_map:
        method_stats["type"] = method_stats["method"].map(method_type_map)

    # Reorder columns
    if method_type_map:
        cols = ["rank", "method", "type", f"{metric}_mean", f"{metric}_std"]
    else:
        cols = ["rank", "method", f"{metric}_mean", f"{metric}_std"]

    ranking = method_stats[cols]

    # Limit to top N
    if top_n:
        ranking = ranking.head(top_n)

    return ranking


def format_table_for_latex(
    df: pd.DataFrame,
    float_format: str = "%.2f",
    bold_best: bool = True,
    caption: str = "",
    label: str = "",
) -> str:
    """
    Format pandas DataFrame for LaTeX publication.

    Parameters
    ----------
    df : pd.DataFrame
        Table to format
    float_format : str
        Format string for floats
    bold_best : bool
        Bold the best value in each column
    caption : str
        LaTeX caption
    label : str
        LaTeX label for referencing

    Returns
    -------
    latex_str : str
        LaTeX table code

    Example
    -------
    >>> latex = format_table_for_latex(summary, caption="Summary Statistics")
    >>> print(latex)
    \\begin{table}[h]
    \\centering
    \\caption{Summary Statistics}
    \\label{tab:summary}
    ...
    \\end{table}
    """
    # Start LaTeX code
    latex_lines = []
    latex_lines.append(r"\begin{table}[h]")
    latex_lines.append(r"\centering")
    if caption:
        latex_lines.append(f"\\caption{{{caption}}}")
    if label:
        latex_lines.append(f"\\label{{tab:{label}}}")

    # Convert DataFrame to LaTeX
    table_str = df.to_latex(
        float_format=float_format,
        escape=False,
        index=True,
        column_format="l" + "r" * len(df.columns),
    )

    # Bold best values if requested
    if bold_best:
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                best_idx = df[col].idxmin()  # Assumes lower is better
                best_val = df.loc[best_idx, col]
                # Replace in LaTeX string
                table_str = table_str.replace(
                    f"{best_val:{float_format}}", f"\\textbf{{{best_val:{float_format}}}}"
                )

    latex_lines.append(table_str)
    latex_lines.append(r"\end{table}")

    return "\n".join(latex_lines)


def format_table_for_markdown(
    df: pd.DataFrame, float_format: str = "%.2f", align: str = "center"
) -> str:
    """
    Format pandas DataFrame for Markdown.

    Parameters
    ----------
    df : pd.DataFrame
        Table to format
    float_format : str
        Format string for floats
    align : str
        Column alignment ('left', 'center', 'right')

    Returns
    -------
    markdown_str : str
        Markdown table

    Example
    -------
    >>> md = format_table_for_markdown(ranking)
    >>> print(md)
    | Rank | Method | gap_mean | gap_std |
    |:----:|:------:|:--------:|:-------:|
    | 1    | FFT    | 3.20     | 1.50    |
    ...
    """
    return df.to_markdown(
        floatfmt=float_format.replace("%", "").replace("f", ""), index=True, tablefmt="github"
    )


def create_performance_gates_table(
    results_df: pd.DataFrame, gates: Dict[str, float], group_by: str = "method"
) -> pd.DataFrame:
    """
    Check which methods pass performance gates.

    Parameters
    ----------
    results_df : pd.DataFrame
        Benchmark results
    gates : dict
        Performance gates, e.g., {'gap': 15.0, 'ds_violation': 0.01}
    group_by : str
        Grouping column

    Returns
    -------
    gate_table : pd.DataFrame
        Table showing pass/fail for each gate

    Example
    -------
    >>> gates = {'gap': 15.0, 'ds_violation': 0.01}
    >>> gate_table = create_performance_gates_table(results, gates)
    >>> print(gate_table)
                    gap_pass  ds_violation_pass  all_pass
    method
    1-FFT-Laplace   True      True              True
    2-Reverse-Time  True      True              True
    B05-Random      False     True              False
    """
    # Aggregate metrics
    method_avg = results_df.groupby(group_by).mean()

    gate_table = pd.DataFrame(index=method_avg.index)

    # Check each gate
    for metric, threshold in gates.items():
        if metric in method_avg.columns:
            # Assume lower is better for most metrics
            gate_table[f"{metric}_pass"] = method_avg[metric] <= threshold
        else:
            gate_table[f"{metric}_pass"] = False

    # Overall pass
    gate_table["all_pass"] = gate_table.all(axis=1)

    return gate_table


def create_convergence_summary(
    history: Dict, time_points: Optional[List[float]] = None
) -> pd.DataFrame:
    """
    Summarize convergence history at specific time points.

    Parameters
    ----------
    history : dict
        Convergence history with 'times', 'objectives', etc.
    time_points : list of float, optional
        Time points to sample (e.g., [10, 30, 60, 120])
        If None, uses quartiles

    Returns
    -------
    summary : pd.DataFrame
        Table with metrics at each time point

    Example
    -------
    >>> summary = create_convergence_summary(history, [10, 60, 120])
    >>> print(summary)
        time  objective  gap    ds_violation  grad_norm
    0   10    1950.2    18.0   0.015         0.032
    1   60    1872.5    13.3   0.008         0.011
    2   120   1850.3    12.0   0.005         0.003
    """
    times = np.array(history["times"])

    if time_points is None:
        # Use quartiles
        time_points = np.percentile(times, [25, 50, 75, 100])

    summary_data = []

    for t in time_points:
        # Find closest time index
        idx = np.argmin(np.abs(times - t))

        row = {"time": times[idx]}

        for key in ["objectives", "gaps", "ds_violations", "grad_norms"]:
            if key in history and len(history[key]) > idx:
                row[key.rstrip("s")] = history[key][idx]

        summary_data.append(row)

    return pd.DataFrame(summary_data)


def create_instance_difficulty_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank instances by difficulty based on average gap across methods.

    Parameters
    ----------
    results_df : pd.DataFrame
        Benchmark results

    Returns
    -------
    difficulty : pd.DataFrame
        Instances ranked by difficulty

    Example
    -------
    >>> difficulty = create_instance_difficulty_table(results)
    >>> print(difficulty)
                instance  avg_gap  std_gap  n_methods  difficulty
    0           had12     3.5      2.1      22         Easy
    1           tai20a    12.3     5.7      22         Medium
    2           tai256c   25.8     8.2      22         Hard
    """
    instance_stats = results_df.groupby("instance")["gap"].agg(["mean", "std", "count"])
    instance_stats.columns = ["avg_gap", "std_gap", "n_methods"]
    instance_stats = instance_stats.sort_values("avg_gap")

    # Classify difficulty
    def classify_difficulty(gap):
        if gap < 10:
            return "Easy"
        if gap < 20:
            return "Medium"
        return "Hard"

    instance_stats["difficulty"] = instance_stats["avg_gap"].apply(classify_difficulty)

    return instance_stats.reset_index()


def style_dataframe(df: pd.DataFrame, highlight_best: bool = True, cmap: str = "RdYlGn_r"):
    """
    Apply styling to DataFrame for Jupyter notebook display.

    Parameters
    ----------
    df : pd.DataFrame
        Table to style
    highlight_best : bool
        Highlight best value in each column
    cmap : str
        Color map name

    Returns
    -------
    styled : pd.io.formats.style.Styler
        Styled DataFrame

    Example
    -------
    >>> styled = style_dataframe(summary)
    >>> styled  # Display in notebook
    """
    styled = df.style

    # Format floats
    styled = styled.format(dict.fromkeys(df.select_dtypes(include=[np.number]).columns, "{:.2f}"))

    # Highlight best
    if highlight_best:
        for col in df.select_dtypes(include=[np.number]).columns:
            styled = styled.highlight_min(subset=[col], color="lightgreen")

    # Color gradient
    styled = styled.background_gradient(
        cmap=cmap, subset=df.select_dtypes(include=[np.number]).columns
    )

    return styled


# ============================================================================
# ADDITIONAL TABLE FUNCTIONS (from VISUALIZATION_SPECS.md)
# ============================================================================


def generate_performance_table(results: Dict, save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Generate performance summary table from benchmark results.

    Parameters
    ----------
    results : dict
        Results dictionary with instance names as keys
        Each value should have: 'size', 'objective', 'best_known', 'gap', 'time', 'ds_violation'
    save_path : str, optional
        Path to save table as CSV

    Returns
    -------
    df : pd.DataFrame
        Performance summary table

    Example
    -------
    >>> results = {
    ...     'had12': {
    ...         'size': 12, 'objective': 1850, 'best_known': 1652,
    ...         'gap': 0.12, 'time': 10.5, 'ds_violation': 5.3e-3
    ...     },
    ...     'tai256c': {
    ...         'size': 256, 'objective': 50123456, 'best_known': 44759294,
    ...         'gap': 0.1199, 'time': 120.3, 'ds_violation': 8.47e-3
    ...     }
    ... }
    >>> df = generate_performance_table(results)
    """
    rows = []
    for instance, res in results.items():
        gap_pct = res["gap"] * 100
        rows.append(
            {
                "Instance": instance,
                "Size": res["size"],
                "Objective": f"{res['objective']:,.0f}",
                "Best Known": f"{res['best_known']:,.0f}",
                "Gap (%)": f"{gap_pct:.2f}",
                "Time (s)": f"{res['time']:.1f}",
                "DS Viol": f"{res['ds_violation']:.2e}",
                "Pass": "Yes" if gap_pct <= 15 and res["ds_violation"] <= 0.01 else "No",
            }
        )

    df = pd.DataFrame(rows)

    print("\n" + "=" * 100)
    print("PERFORMANCE SUMMARY")
    print("=" * 100)
    print(df.to_string(index=False))
    print("=" * 100 + "\n")

    if save_path:
        df.to_csv(save_path, index=False)
        print(f"[OK] Performance summary saved to: {save_path}\n")

    return df


def generate_ablation_table(
    ablation_results: Dict[str, Dict], save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Generate ablation study table comparing configurations.

    Shows performance with/without each method/component.

    Parameters
    ----------
    ablation_results : dict
        Keys: configuration names
        Values: dicts with 'objective', 'gap', 'time', 'speedup' (optional)

    save_path : str, optional
        Path to save table as CSV

    Returns
    -------
    df : pd.DataFrame
        Ablation study table

    Example
    -------
    >>> ablation_results = {
    ...     'Baseline': {'objective': 2100, 'gap': 0.2711, 'time': 15.2, 'speedup': 1.0},
    ...     '+ Momentum': {'objective': 1950, 'gap': 0.18, 'time': 12.5, 'speedup': 1.22},
    ...     '+ FFT': {'objective': 1900, 'gap': 0.1508, 'time': 8.3, 'speedup': 1.83},
    ...     'Full (all)': {'objective': 1820, 'gap': 0.1017, 'time': 9.8, 'speedup': 1.55}
    ... }
    >>> df = generate_ablation_table(ablation_results)
    """
    rows = []
    for config_name, metrics in ablation_results.items():
        gap_pct = metrics["gap"] * 100
        speedup = metrics.get("speedup", 1.0)

        rows.append(
            {
                "Configuration": config_name,
                "Objective": f"{metrics['objective']:,.0f}",
                "Gap (%)": f"{gap_pct:.2f}",
                "Time (s)": f"{metrics['time']:.1f}",
                "Speedup": f"{speedup:.2f}×",
            }
        )

    df = pd.DataFrame(rows)

    print("\n" + "=" * 90)
    print("ABLATION STUDY")
    print("=" * 90)
    print(df.to_string(index=False))
    print("=" * 90 + "\n")

    if save_path:
        df.to_csv(save_path, index=False)
        print(f"[OK] Ablation table saved to: {save_path}\n")

    return df
