"""
Phase 2 Results Analysis

Analyzes Phase 2 evaluation results to identify:
1. Librex.Meta's ranking across scenarios
2. Scenarios where Librex.Meta excels
3. Performance patterns and insights
"""

import pandas as pd
import numpy as np

# Load results
df = pd.read_csv('results/phase2/phase2_results_summary.csv')

print("="*70)
print("Phase 2 Results Analysis")
print("="*70)

# 1. Overall ranking by scenario
print("\n" + "="*70)
print("1. Librex.Meta Ranking by Scenario")
print("="*70)

scenarios = df['Scenario'].unique()

for scenario in scenarios:
    scenario_df = df[df['Scenario'] == scenario].copy()

    # Rank by regret (lower is better)
    scenario_df = scenario_df.sort_values('Avg Regret')
    scenario_df['Rank'] = range(1, len(scenario_df) + 1)

    # Find Librex.Meta optimal
    ml_optimal = scenario_df[scenario_df['Method'] == 'Librex.Meta (optimal)']

    if len(ml_optimal) > 0:
        rank = ml_optimal['Rank'].values[0]
        regret = ml_optimal['Avg Regret'].values[0]
        top1 = ml_optimal['Top-1 Acc'].values[0]

        # Get best method
        best = scenario_df.iloc[0]

        print(f"\n{scenario}:")
        print(f"  Librex.Meta (optimal): Rank {rank}/{len(scenario_df)}")
        print(f"    Regret: {regret:.4f}, Top-1: {top1:.1%}")
        print(f"  Best method: {best['Method']}")
        print(f"    Regret: {best['Avg Regret']:.4f}, Top-1: {best['Top-1 Acc']:.1%}")

        if rank == 1:
            print(f"  ✅ Librex.Meta WINS!")
        elif rank <= 2:
            print(f"  🎯 Librex.Meta COMPETITIVE (top 2)")
        elif rank <= 3:
            print(f"  ⚠️ Librex.Meta DECENT (top 3)")
        else:
            print(f"  ❌ Librex.Meta POOR (rank {rank})")

# 2. Overall performance across all scenarios
print("\n" + "="*70)
print("2. Average Performance Across All Scenarios")
print("="*70)

# Filter to only optimal configs
ml_df = df[df['Method'].str.contains('Librex.Meta')].copy()

# Group by method
agg_df = df.groupby('Method').agg({
    'Avg Regret': 'mean',
    'Top-1 Acc': 'mean',
    'Top-3 Acc': 'mean',
    'Sel Time (ms)': 'mean'
}).round(4)

agg_df = agg_df.sort_values('Avg Regret')

print("\n" + agg_df.to_string())

# 3. Scenarios where Librex.Meta excels
print("\n" + "="*70)
print("3. Scenarios Where Librex.Meta Excels")
print("="*70)

wins = []
competitive = []
poor = []

for scenario in scenarios:
    scenario_df = df[df['Scenario'] == scenario].copy()
    scenario_df = scenario_df.sort_values('Avg Regret')
    scenario_df['Rank'] = range(1, len(scenario_df) + 1)

    ml_optimal = scenario_df[scenario_df['Method'] == 'Librex.Meta (optimal)']

    if len(ml_optimal) > 0:
        rank = ml_optimal['Rank'].values[0]

        if rank == 1:
            wins.append(scenario)
        elif rank <= 2:
            competitive.append(scenario)
        else:
            poor.append(scenario)

print(f"\n✅ WINS (Rank 1): {wins}")
print(f"🎯 COMPETITIVE (Rank 2-3): {competitive}")
print(f"❌ POOR (Rank 4+): {poor}")

# 4. Speed comparison
print("\n" + "="*70)
print("4. Selection Speed Comparison")
print("="*70)

speed_df = df.groupby('Method')['Sel Time (ms)'].mean().sort_values()
print("\n" + speed_df.to_string())

# Find Librex.Meta speedup
ml_speed = speed_df['Librex.Meta (optimal)']
satzilla_speed = speed_df['SATzilla']
autofolio_speed = speed_df['AutoFolio']

print(f"\nLibrex.Meta speedup:")
print(f"  vs SATzilla: {satzilla_speed / ml_speed:.0f}x faster")
print(f"  vs AutoFolio: {autofolio_speed / ml_speed:.0f}x faster")

# 5. Optimal vs Default comparison
print("\n" + "="*70)
print("5. Optimal vs Default Configuration")
print("="*70)

ml_only = df[df['Method'].str.contains('Librex.Meta')].copy()

comparison = ml_only.groupby(['Scenario', 'Method']).agg({
    'Avg Regret': 'mean',
    'Top-1 Acc': 'mean'
}).reset_index()

comparison_pivot = comparison.pivot(index='Scenario', columns='Method', values=['Avg Regret', 'Top-1 Acc'])

print("\n" + comparison_pivot.to_string())

# Calculate improvements
improvements = []
for scenario in scenarios:
    scenario_df = ml_only[ml_only['Scenario'] == scenario]

    default = scenario_df[scenario_df['Method'] == 'Librex.Meta (default)']
    optimal = scenario_df[scenario_df['Method'] == 'Librex.Meta (optimal)']

    if len(default) > 0 and len(optimal) > 0:
        default_regret = default['Avg Regret'].values[0]
        optimal_regret = optimal['Avg Regret'].values[0]

        default_top1 = default['Top-1 Acc'].values[0]
        optimal_top1 = optimal['Top-1 Acc'].values[0]

        regret_improvement = (default_regret - optimal_regret) / default_regret * 100
        top1_improvement = (optimal_top1 - default_top1) / default_top1 * 100

        improvements.append({
            'Scenario': scenario,
            'Regret Improvement %': regret_improvement,
            'Top-1 Improvement %': top1_improvement
        })

improvements_df = pd.DataFrame(improvements)
print("\n" + "="*70)
print("Optimal Configuration Impact:")
print("="*70)
print("\n" + improvements_df.to_string(index=False))

print(f"\nAverage improvements:")
print(f"  Regret: {improvements_df['Regret Improvement %'].mean():.1f}%")
print(f"  Top-1 Accuracy: {improvements_df['Top-1 Improvement %'].mean():.1f}%")

print("\n" + "="*70)
print("✓ Analysis Complete!")
print("="*70)
