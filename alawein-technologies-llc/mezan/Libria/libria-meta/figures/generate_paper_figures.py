"""
Generate Figures for Librex.Meta Paper

Creates three key figures from Phase 2 and ablation study results:
- Figure 2: Per-scenario performance (grouped bar chart)
- Figure 3: Speed-accuracy Pareto frontier (scatter plot)
- Figure 4: Hyperparameter sensitivity (2×2 grid of line plots)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set style
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# Create output directory
output_dir = Path('figures/output')
output_dir.mkdir(parents=True, exist_ok=True)

#==============================================================================
# Figure 2: Per-Scenario Performance (Grouped Bar Chart)
#==============================================================================

def figure2_scenario_performance():
    """Generate grouped bar chart showing regret by scenario and method."""

    # Load Phase 2 results
    df = pd.read_csv('results/phase2/phase2_results_summary.csv')

    # Pivot to get methods as columns, scenarios as rows
    pivot = df.pivot(index='Scenario', columns='Method', values='Avg Regret')

    # Reorder scenarios by Librex.Meta (optimal) performance
    ml_optimal = pivot['Librex.Meta (optimal)']
    pivot = pivot.loc[ml_optimal.sort_values().index]

    # Select key methods for clarity
    methods_to_plot = [
        'Librex.Meta (optimal)',
        'SATzilla',
        'AutoFolio',
        'SMAC',
        'Hyperband'
    ]
    pivot = pivot[methods_to_plot]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot grouped bars
    pivot.plot(kind='bar', ax=ax, width=0.8)

    # Styling
    ax.set_xlabel('Scenario', fontweight='bold')
    ax.set_ylabel('Average Regret', fontweight='bold')
    ax.set_title('Figure 2: Per-Scenario Performance Comparison', fontweight='bold', pad=15)
    ax.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')

    # Highlight Librex.Meta wins
    scenarios = pivot.index.tolist()
    for i, scenario in enumerate(scenarios):
        best_method = pivot.loc[scenario].idxmin()
        if best_method == 'Librex.Meta (optimal)':
            # Add star annotation
            ax.text(i, pivot.loc[scenario].max() + 0.01, '★',
                   ha='center', va='bottom', fontsize=16, color='gold')

    plt.tight_layout()
    plt.savefig(output_dir / 'figure2_scenario_performance.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'figure2_scenario_performance.pdf', bbox_inches='tight')
    print(f"✓ Figure 2 saved to {output_dir}")
    plt.close()

#==============================================================================
# Figure 3: Speed-Accuracy Pareto Frontier (Scatter Plot)
#==============================================================================

def figure3_pareto_frontier():
    """Generate scatter plot showing speed vs accuracy trade-off."""

    # Load Phase 2 results
    df = pd.read_csv('results/phase2/phase2_results_summary.csv')

    # Aggregate by method
    agg = df.groupby('Method').agg({
        'Avg Regret': 'mean',
        'Sel Time (ms)': 'mean'
    }).reset_index()

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Define colors for each method
    colors = {
        'Librex.Meta (optimal)': '#e74c3c',  # Red
        'Librex.Meta (default)': '#e67e73',  # Light red
        'SATzilla': '#3498db',  # Blue
        'AutoFolio': '#2ecc71',  # Green
        'SMAC': '#f39c12',  # Orange
        'Hyperband': '#95a5a6',  # Gray
        'BOHB': '#7f8c8d'  # Dark gray
    }

    # Plot each method
    for _, row in agg.iterrows():
        method = row['Method']
        x = row['Sel Time (ms)']
        y = row['Avg Regret']

        # Size: highlight Librex.Meta
        size = 200 if 'Librex.Meta' in method else 100

        ax.scatter(x, y, s=size, alpha=0.7, color=colors.get(method, '#34495e'),
                  label=method, edgecolors='black', linewidth=1.5)

    # Log scale for x-axis (selection time varies 0.03ms to 254ms)
    ax.set_xscale('log')

    # Add Pareto frontier line (connect dominant methods)
    pareto_methods = ['Hyperband', 'Librex.Meta (optimal)', 'SATzilla']
    pareto_data = agg[agg['Method'].isin(pareto_methods)].sort_values('Sel Time (ms)')
    ax.plot(pareto_data['Sel Time (ms)'], pareto_data['Avg Regret'],
           'k--', alpha=0.3, linewidth=1, label='Pareto frontier')

    # Annotate Librex.Meta (optimal)
    ml_row = agg[agg['Method'] == 'Librex.Meta (optimal)'].iloc[0]
    ax.annotate('1664× faster\nthan SATzilla',
               xy=(ml_row['Sel Time (ms)'], ml_row['Avg Regret']),
               xytext=(1, 0.08),
               arrowprops=dict(arrowstyle='->', lw=1.5),
               fontsize=10, ha='left', bbox=dict(boxstyle='round,pad=0.5',
                                                  facecolor='yellow', alpha=0.7))

    # Shading: real-time region (<1ms)
    ax.axvspan(0.01, 1, alpha=0.1, color='green', label='Real-time region (<1ms)')

    # Styling
    ax.set_xlabel('Selection Time (ms, log scale)', fontweight='bold')
    ax.set_ylabel('Average Regret', fontweight='bold')
    ax.set_title('Figure 3: Speed-Accuracy Trade-off (Pareto Frontier)', fontweight='bold', pad=15)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_dir / 'figure3_pareto_frontier.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'figure3_pareto_frontier.pdf', bbox_inches='tight')
    print(f"✓ Figure 3 saved to {output_dir}")
    plt.close()

#==============================================================================
# Figure 4: Hyperparameter Sensitivity (2×2 Grid)
#==============================================================================

def figure4_hyperparameter_sensitivity():
    """Generate 2×2 grid of line plots showing hyperparameter sensitivity."""

    # Load ablation study results
    ablation_files = {
        'n_clusters': 'results/ablation_real/ablation_n_clusters_real.csv',
        'ucb_c': 'results/ablation_real/ablation_ucb_constant_real.csv',
        'n_rounds': 'results/ablation_real/ablation_n_tournament_rounds_real.csv',
        'elo_k': 'results/ablation_real/ablation_elo_k_real.csv'
    }

    # Create 2×2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # Colors for scenarios
    scenario_colors = {'SAT11-HAND': '#3498db', 'CSP-2010': '#e74c3c'}

    # Subplot A: n_clusters
    if Path(ablation_files['n_clusters']).exists():
        df = pd.read_csv(ablation_files['n_clusters'])
        ax = axes[0]

        for scenario in df['scenario'].unique():
            data = df[df['scenario'] == scenario]
            ax.plot(data['n_clusters'], data['avg_regret'],
                   marker='o', linewidth=2, markersize=6,
                   label=scenario, color=scenario_colors.get(scenario, 'gray'))

        # Highlight optimal k=3
        ax.axvline(3, color='green', linestyle='--', alpha=0.5, linewidth=2)
        ax.text(3, ax.get_ylim()[1] * 0.95, 'Optimal', ha='center',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))

        ax.set_xlabel('Number of Clusters (k)', fontweight='bold')
        ax.set_ylabel('Average Regret', fontweight='bold')
        ax.set_title('(A) n_clusters: Significant Impact', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Subplot B: ucb_c
    if Path(ablation_files['ucb_c']).exists():
        df = pd.read_csv(ablation_files['ucb_c'])
        ax = axes[1]

        for scenario in df['scenario'].unique():
            data = df[df['scenario'] == scenario]
            ax.plot(data['ucb_c'], data['avg_regret'],
                   marker='o', linewidth=2, markersize=6,
                   label=scenario, color=scenario_colors.get(scenario, 'gray'))

        ax.text(1.0, ax.get_ylim()[1] * 0.95, 'NO IMPACT\n(flat line)',
               ha='center', fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.7))

        ax.set_xlabel('UCB Constant (c)', fontweight='bold')
        ax.set_ylabel('Average Regret', fontweight='bold')
        ax.set_title('(B) ucb_c: No Impact', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Subplot C: n_rounds
    if Path(ablation_files['n_rounds']).exists():
        df = pd.read_csv(ablation_files['n_rounds'])
        ax = axes[2]

        for scenario in df['scenario'].unique():
            data = df[df['scenario'] == scenario]
            ax.plot(data['n_tournament_rounds'], data['avg_regret'],
                   marker='o', linewidth=2, markersize=6,
                   label=scenario, color=scenario_colors.get(scenario, 'gray'))

        ax.text(7.5, ax.get_ylim()[1] * 0.95, 'NO IMPACT\n(flat line)',
               ha='center', fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.7))

        ax.set_xlabel('Tournament Rounds', fontweight='bold')
        ax.set_ylabel('Average Regret', fontweight='bold')
        ax.set_title('(C) n_tournament_rounds: No Impact', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Subplot D: elo_k
    if Path(ablation_files['elo_k']).exists():
        df = pd.read_csv(ablation_files['elo_k'])
        ax = axes[3]

        for scenario in df['scenario'].unique():
            data = df[df['scenario'] == scenario]
            ax.plot(data['elo_k'], data['avg_regret'],
                   marker='o', linewidth=2, markersize=6,
                   label=scenario, color=scenario_colors.get(scenario, 'gray'))

        # Highlight avoid K=8, 16
        ax.axvspan(7, 17, alpha=0.2, color='red', label='Avoid')
        ax.text(32, ax.get_ylim()[1] * 0.95, 'K=32 optimal', ha='center',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))

        ax.set_xlabel('Elo K (update rate)', fontweight='bold')
        ax.set_ylabel('Average Regret', fontweight='bold')
        ax.set_title('(D) elo_k: Avoid 8 and 16', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Figure 4: Hyperparameter Sensitivity Analysis',
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'figure4_hyperparameter_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'figure4_hyperparameter_sensitivity.pdf', bbox_inches='tight')
    print(f"✓ Figure 4 saved to {output_dir}")
    plt.close()

#==============================================================================
# Main
#==============================================================================

if __name__ == '__main__':
    print("="*70)
    print("Generating Paper Figures")
    print("="*70)

    print("\nGenerating Figure 2: Per-Scenario Performance...")
    figure2_scenario_performance()

    print("\nGenerating Figure 3: Speed-Accuracy Pareto Frontier...")
    figure3_pareto_frontier()

    print("\nGenerating Figure 4: Hyperparameter Sensitivity...")
    figure4_hyperparameter_sensitivity()

    print("\n" + "="*70)
    print("✓ All figures generated successfully!")
    print(f"✓ Saved to: {output_dir.absolute()}")
    print("="*70)
