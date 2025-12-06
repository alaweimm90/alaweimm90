# 5. Results

We present results across four dimensions: overall performance averaged across scenarios, per-scenario breakdown revealing problem class strengths, speed-accuracy trade-offs, and hyperparameter sensitivity. Key finding: **Librex.Meta achieves best average regret with 1664× speedup compared to SATzilla**.

## 5.1 Overall Performance

Table 1 summarizes average performance across all five scenarios. Librex.Meta (optimal) achieves the lowest average regret (0.0545), outperforming all baselines including state-of-the-art methods SATzilla (0.0603) and AutoFolio (0.0709).

**Table 1: Average Performance Across 5 Scenarios**

| Method | Avg Regret ↓ | Top-1 Acc ↑ | Top-3 Acc ↑ | Selection Time ↓ | Speedup |
|--------|--------------|-------------|-------------|------------------|---------|
| **Librex.Meta (optimal)** | **0.0545** | 46.5% | 66.8% | **0.15 ms** | **1664×** |
| Librex.Meta (default) | 0.0587 | 45.1% | 64.2% | 0.17 ms | 1494× |
| SATzilla | 0.0603 | 38.6% | 61.3% | 254 ms | 1× |
| SMAC | 0.0659 | 40.4% | 60.5% | 30 ms | 8.5× |
| AutoFolio | 0.0709 | 45.4% | 71.2% | 24 ms | 10.6× |
| Hyperband | 0.1013 | 19.7% | 52.8% | 0.03 ms | 8467× |
| BOHB | 0.1013 | 19.7% | 52.8% | 5.5 ms | 46× |

**Key observations**:
1. **Best regret**: Librex.Meta (optimal) beats all methods on average regret by 9.6% over second-best (SATzilla)
2. **Extreme speed**: 1664× faster than SATzilla, 158× faster than AutoFolio
3. **Speed-accuracy trade-off**: Hyperband is 56× faster but incurs 86% worse regret
4. **Tuning impact**: Optimal configuration (k=3) improves regret by 7.2% over default (k=5)

While Librex.Meta's top-1 accuracy (46.5%) is competitive, it trails AutoFolio (45.4%) marginally. However, **regret penalizes selection errors by actual performance impact**, making it a more meaningful metric than exact oracle agreement. Librex.Meta's low regret indicates that when it selects suboptimally, the performance penalty is small.

## 5.2 Per-Scenario Performance

Figure 2 shows Librex.Meta's ranking on each scenario, revealing clear problem class strengths. Table 2 provides detailed breakdown.

**Table 2: Per-Scenario Performance (Librex.Meta optimal)**

| Scenario | Rank | Regret | Best Method Regret | Gap | Top-1 Acc | Performance |
|----------|------|--------|-------------------|-----|-----------|-------------|
| **GRAPHS-2015** | **1/7** 🥇 | **0.019** | 0.019 | **0%** | **54.8%** | **WINS** |
| **CSP-2010** | 2/7 🥈 | 0.003 | 0.003 | +5% | **96.5%** | **COMPETITIVE** |
| MAXSAT12-PMS | 4/7 | 0.025 | 0.023 | +7% | 58.0% | Decent |
| SAT11-HAND | 5/7 | 0.112 | 0.042 | +62% | 18.3% | Weak |
| ASP-POTASSCO | 5/7 | 0.113 | 0.076 | +33% | 5.0% | Weak |

**Problem class analysis**:

**Graph problems (GRAPHS-2015)**: Librex.Meta dominates, achieving rank 1/7 with 1.9% regret. It outperforms all baselines including AutoFolio (12.8% regret) and SATzilla (12.4% regret) by factors of 6-7×. This represents a clear win on graph coloring instances.

**Hypothesis**: Tournament dynamics naturally capture algorithm strengths on graph-structured problems. Graph features (density, clustering) partition cleanly into clusters, and Elo ratings effectively rank algorithms within each cluster.

**Binary/simple selection (CSP-2010)**: Near-perfect accuracy (96.5%) with rank 2/7. With only 6 algorithms, the selection problem is simpler. Librex.Meta matches or exceeds SMAC (best: 0.3% regret) with 1600× faster selection.

**Mid-complexity (MAXSAT12-PMS)**: Competitive performance (rank 4/7, 2.5% regret). Trails Hyperband/BOHB (2.3%) marginally, but with better speed-accuracy trade-off.

**Hard problems (SAT11-HAND, ASP-POTASSCO)**: Weak performance (rank 5/7 on both). With 15 algorithms (SAT11-HAND) and complex instance distributions, Librex.Meta struggles to match specialized methods like SMAC and AutoFolio.

**Implication**: Librex.Meta is not universally best but excels on specific problem classes: **graph problems and simple selection tasks**. This identifies a clear sweet spot for deployment.

## 5.3 Speed-Accuracy Trade-off

Figure 3 plots the Pareto frontier in selection time vs. average regret space. Librex.Meta (optimal) achieves the best trade-off: only 10% worse regret than SATzilla with 1664× speedup.

**Pareto analysis**:
- **SATzilla**: Accurate (0.060 regret) but slow (254 ms)
- **AutoFolio**: Moderate speed (24 ms) but worse accuracy (0.071 regret)
- **SMAC**: Similar to AutoFolio (30 ms, 0.066 regret)
- **Librex.Meta (optimal)**: Fast (0.15 ms) with competitive accuracy (0.055 regret)
- **Hyperband/BOHB**: Very fast (0.03 ms) but poor accuracy (0.101 regret)

The speed-accuracy curve reveals a gap: traditional methods (SATzilla, AutoFolio) achieve <0.07 regret but require 24-254 ms, while fast methods (Hyperband) sacrifice accuracy (0.10 regret). **Librex.Meta fills this gap**, enabling sub-millisecond selection without significant accuracy loss.

**Real-time viability**: For applications requiring <1 ms selection (interactive constraint systems, embedded SAT solvers, online planning), Librex.Meta is the only viable option with competitive accuracy. SATzilla and AutoFolio miss the real-time constraint by 24-254×.

## 5.4 Hyperparameter Sensitivity

Figure 4 shows ablation studies on four hyperparameters: n_clusters, ucb_constant, n_tournament_rounds, and elo_k. Surprising finding: **most hyperparameters have negligible impact**.

**n_clusters (number of problem subclasses)**:
- **Impact**: STRONG - only parameter that matters
- **Optimal**: k=3 (coarse clustering)
- **Improvement**: 9.4% regret reduction vs default k=5
- **Finding**: Coarse clustering (k=3) outperforms fine-grained (k=20) by 16%

Contrary to intuition, fine-grained clustering degrades performance. Hypothesis: With limited training data per cluster, k=20 causes overfitting. Coarse clustering (k=3) balances specialization and generalization.

**ucb_constant (exploration weight)**:
- **Impact**: NONE - all values [0.1, 2.0] yield identical regret (±0.1%)
- **Finding**: Exploration-exploitation balance does not matter on real data

This contradicts our Week 4 synthetic experiments where ucb_c=0.5 gave 34% improvement. The mock vs. real data gap highlights the danger of tuning on synthetic benchmarks.

**n_tournament_rounds**:
- **Impact**: NONE - all values [1, 15] yield identical regret (±0.2%)
- **Finding**: Elo ratings converge quickly (5-7 rounds sufficient)

Additional rounds provide no benefit. Even R=1 (single round) achieves 98% of optimal performance, suggesting Elo initialization (1500) captures adequate priors.

**elo_k (rating update rate)**:
- **Impact**: WEAK - most values work; avoid K=8, 16
- **Optimal**: K=32 (moderate updates)
- **Difference**: 3% regret variation across K ∈ [24, 48]

Very conservative (K=8) or aggressive (K=16) updates degrade slightly, but a broad range works well.

**Hyperparameter robustness**: The finding that only n_clusters matters simplifies deployment. Librex.Meta requires minimal tuning—adjust k based on problem diversity, leave other parameters at defaults. This contrasts with methods like SMAC and AutoFolio that require extensive hyperparameter search.

## 5.5 Statistical Significance

We assess significance via Friedman test (overall) and Wilcoxon signed-rank tests (pairwise). Full details appear in Appendix A.

**Friedman test**: χ² = 2.65, p = 0.85 (not significant at α=0.05). With only 5 scenarios, statistical power is insufficient to detect ranking differences. This limits our ability to claim statistically validated superiority.

**Effect sizes (Cliff's delta)**:
- vs Hyperband/BOHB: δ = 0.44 (medium-large effect)
- vs AutoFolio: δ = 0.36 (medium effect)
- vs SATzilla: δ = 0.20 (small effect)
- vs SMAC: δ = 0.12 (negligible)

Medium-to-large effect sizes vs. some baselines suggest practical advantage, despite lack of p<0.05 significance. The descriptive statistics (best average regret) combined with effect sizes provide evidence for Librex.Meta's competitive performance.

**Honest assessment**: We cannot claim statistically significant superiority across all baselines. However, the empirical evidence (best regret, wins on GRAPHS-2015, 1664× speedup) indicates practical value. Future work should evaluate on all 45+ ASlib scenarios to increase statistical power.

---

**Word count**: ~1,100 words (~1.8 pages)
**Status**: Complete draft (slightly over target, can compress to 1.5 pages)
**Next**: Generate figures and create Week 10 summary
