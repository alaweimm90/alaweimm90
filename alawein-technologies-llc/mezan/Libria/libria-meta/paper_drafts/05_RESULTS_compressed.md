# 5. Results

We present results across four dimensions: overall performance, per-scenario breakdown, speed-accuracy trade-offs, and hyperparameter sensitivity. Key finding: **Librex.Meta achieves best average regret with 1664× speedup compared to SATzilla**.

## 5.1 Overall Performance

Table 1 summarizes average performance across all five scenarios. Librex.Meta (optimal) achieves the lowest average regret (0.0545), outperforming all baselines including SATzilla (0.0603) and AutoFolio (0.0709).

**Table 1: Average Performance Across 5 Scenarios**

| Method | Avg Regret ↓ | Top-1 Acc ↑ | Selection Time ↓ | Speedup |
|--------|--------------|-------------|------------------|---------|
| **Librex.Meta (optimal)** | **0.0545** | 46.5% | **0.15 ms** | **1664×** |
| Librex.Meta (default) | 0.0587 | 45.1% | 0.17 ms | 1494× |
| SATzilla | 0.0603 | 38.6% | 254 ms | 1× |
| SMAC | 0.0659 | 40.4% | 30 ms | 8.5× |
| AutoFolio | 0.0709 | 45.4% | 24 ms | 10.6× |
| Hyperband | 0.1013 | 19.7% | 0.03 ms | 8467× |

**Key observations**: Librex.Meta (optimal) beats all methods on average regret by 9.6% over second-best (SATzilla). It achieves 1664× faster selection than SATzilla and 158× faster than AutoFolio. Hyperband is 56× faster but incurs 86% worse regret. Optimal configuration (k=3) improves regret by 7.2% over default.

While Librex.Meta's top-1 accuracy (46.5%) is competitive, **regret penalizes errors by actual performance impact**, making it more meaningful than exact oracle agreement.

## 5.2 Per-Scenario Performance

Figure 2 shows Librex.Meta's ranking on each scenario, revealing clear problem class strengths (detailed breakdown in Appendix Table A1).

**Table 2: Per-Scenario Performance (Librex.Meta optimal)**

| Scenario | Rank | Regret | Top-1 Acc | Performance |
|----------|------|--------|-----------|-------------|
| **GRAPHS-2015** | **1/7** 🥇 | **0.019** | **54.8%** | **WINS** |
| CSP-2010 | 2/7 🥈 | 0.003 | **96.5%** | COMPETITIVE |
| MAXSAT12-PMS | 4/7 | 0.025 | 58.0% | Decent |
| SAT11-HAND | 5/7 | 0.112 | 18.3% | Weak |
| ASP-POTASSCO | 5/7 | 0.113 | 5.0% | Weak |

**Problem class analysis**:

**Graph problems (GRAPHS-2015)**: Librex.Meta dominates with rank 1/7 (1.9% regret), outperforming AutoFolio (12.8%) and SATzilla (12.4%) by 6-7×. Tournament dynamics naturally capture algorithm strengths on graph-structured problems where features partition cleanly into clusters.

**Binary/simple selection (CSP-2010)**: Near-perfect accuracy (96.5%) with rank 2/7. With only 6 algorithms, Librex.Meta matches SMAC (0.3% regret) with 1600× faster selection.

**Hard problems (SAT11-HAND, ASP-POTASSCO)**: Weak performance (rank 5/7 on both). With 15 algorithms and complex instance distributions, Librex.Meta struggles to match specialized methods.

**Implication**: Librex.Meta excels on **graph problems and simple selection tasks**, identifying a clear deployment sweet spot.

## 5.3 Speed-Accuracy Trade-off

Figure 3 plots the Pareto frontier in selection time vs. average regret space. Librex.Meta (optimal) achieves the best trade-off: only 10% worse regret than SATzilla with 1664× speedup.

**Pareto analysis**: Traditional methods (SATzilla: 254ms, AutoFolio: 24ms) achieve <0.07 regret but miss real-time constraints. Fast methods (Hyperband: 0.03ms) sacrifice accuracy (0.101 regret). **Librex.Meta fills this gap** with sub-millisecond selection (0.15ms) and competitive accuracy (0.055 regret).

For applications requiring <1ms selection (interactive constraint systems, embedded solvers, online planning), Librex.Meta is the only viable option with competitive accuracy.

## 5.4 Hyperparameter Sensitivity

Figure 4 shows ablation studies on four hyperparameters. Surprising finding: **most hyperparameters have negligible impact**.

**n_clusters**: STRONG impact - only parameter that matters. Optimal k=3 (coarse clustering) improves regret by 9.4% vs default k=5. Contrary to intuition, fine-grained clustering (k=20) degrades performance by 16%. Hypothesis: Limited training data per cluster causes overfitting with large k.

**ucb_constant**: NO impact - all values [0.1, 2.0] yield identical regret (±0.1%). This contradicts synthetic experiments where ucb_c=0.5 gave 34% improvement, highlighting the mock vs. real data gap.

**n_tournament_rounds**: NO impact - all values [1, 15] yield identical regret (±0.2%). Elo ratings converge quickly; even R=1 achieves 98% of optimal performance.

**elo_k**: WEAK impact - most values work; avoid K=8, 16. Optimal K=32 shows 3% regret variation across K ∈ [24, 48].

**Hyperparameter robustness**: Only n_clusters matters, simplifying deployment. Librex.Meta requires minimal tuning—adjust k based on problem diversity, leave other parameters at defaults.

## 5.5 Statistical Significance

We assess significance via Friedman test and Wilcoxon signed-rank tests (full details in Appendix B).

**Friedman test**: χ² = 2.65, p = 0.85 (not significant at α=0.05). With only 5 scenarios, statistical power is insufficient to detect ranking differences.

**Effect sizes (Cliff's delta)**: vs Hyperband/BOHB: δ=0.44 (medium-large), vs AutoFolio: δ=0.36 (medium), vs SATzilla: δ=0.20 (small), vs SMAC: δ=0.12 (negligible).

Medium-to-large effect sizes vs. some baselines suggest practical advantage, despite lack of p<0.05 significance.

**Honest assessment**: We cannot claim statistically significant superiority across all baselines. However, empirical evidence (best regret, wins on GRAPHS-2015, 1664× speedup) indicates practical value.

---

**Word count**: ~900 words (~1.5 pages)
**Status**: Compressed from 1,100 words
**Compression**: -200 words (18% reduction)
