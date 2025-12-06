# 2. Related Work

Librex.Meta builds on three research threads: algorithm selection, meta-learning for hyperparameter optimization, and tournament-based learning systems.

## 2.1 Algorithm Selection

**Classical approaches.** The algorithm selection problem was formalized by Rice (1976). Modern approaches leverage machine learning to predict algorithm performance from instance features.

**SATzilla** (Xu et al., 2008, 2012) pioneered regression-based algorithm selection for SAT solving, training cost-sensitive models to predict solver runtimes from CNF features. While highly accurate (winning multiple SAT competitions), SATzilla's selection overhead averages 254ms, limiting real-time applicability.

**AutoFolio** (Lindauer et al., 2015) extends this paradigm by automatically configuring the selection pipeline itself using AutoML, yielding strong performance across diverse domains but requiring 24ms selection time.

**Other approaches** include schedule-based methods (3S, ISAC) that run solvers in learned sequences, but introduce overhead incompatible with sub-millisecond requirements.

**ASlib** (Bischl et al., 2016) established a standardized benchmark suite with 45+ scenarios across SAT, CSP, QBF, ASP, and other domains, enabling reproducible comparison. Our evaluation uses 5 diverse ASlib scenarios.

**Limitation.** State-of-the-art methods achieve strong accuracy but high latency (24-254ms). Librex.Meta targets sub-millisecond selection through tournament-based meta-learning.

## 2.2 Meta-Learning for Algorithm Selection

**Hyperparameter optimization (HPO)** methods treat algorithm selection as black-box optimization.

**SMAC** (Hutter et al., 2011) builds random forest surrogates of algorithm performance, excelling at configuration but requiring 30ms+ for selection.

**Hyperband** (Li et al., 2017) allocates resources adaptively using successive halving, achieving fast selection (0.03ms) but poor accuracy on algorithm selection (top-1: 19.7% in our experiments) due to insufficient exploitation of problem structure.

**BOHB** (Falkner et al., 2018) combines Bayesian optimization with Hyperband using kernel density estimators, but still struggles on algorithm selection (19.7% top-1 accuracy).

**Auto-sklearn** (Feurer et al., 2015) and Auto-WEKA (Kotthoff et al., 2017) extend meta-learning to full ML pipeline construction but target offline optimization, not real-time selection.

**Gap.** Existing approaches either sacrifice speed (SMAC, Auto-sklearn) or accuracy (Hyperband, BOHB). Librex.Meta achieves both through tournament-based learning with pre-computed Elo ratings.

## 2.3 Tournament-Based Learning

**Elo rating system** (Elo, 1978) ranks chess players based on match outcomes, with ratings updated after games based on expected vs actual performance. Elo ratings handle partial information and converge quickly.

**TrueSkill** (Herbrich et al., 2007) extends Elo using Bayesian inference with uncertainty estimates.

**Applications in machine learning** include tournament selection in evolutionary algorithms and active learning query selection. However, we are unaware of prior work applying Swiss-system tournaments and Elo ratings to algorithm selection meta-learning.

**Our contribution.** Librex.Meta introduces tournament-based meta-learning to algorithm selection, using Swiss-system tournaments for efficient ranking and dual Elo ratings (global + cluster-specific) for specialized performance tracking.

## 2.4 Clustering for Problem Space Partitioning

**Instance space analysis** (Smith-Miles & Bowly, 2015) studies relationships between problem features and algorithm performance, motivating clustering as a specialization mechanism.

**Feature clustering for algorithm selection** appears in several systems. Leyton-Brown et al. (2009) cluster SAT instances and train specialized models per cluster.

**Our approach.** Librex.Meta uses KMeans clustering with cluster-specific Elo ratings to capture specialized algorithm performance. Unlike prior work that trains separate models, we specialize Elo ratings, reducing complexity. We also discover that coarse clustering (k=3) outperforms fine-grained partitioning (k=20).

## 2.5 Positioning

Librex.Meta occupies a unique position in the algorithm selection landscape. Compared to classical methods (SATzilla, AutoFolio), we sacrifice ~10% accuracy for 1600× speedup. Compared to fast baselines (Hyperband), we improve accuracy dramatically (46.5% vs 19.7% top-1) with similar latency.

Table 1 summarizes key differences:

| Method | Approach | Selection Time | Regret | Novelty |
|--------|----------|----------------|--------|---------|
| SATzilla | Regression + pre-solvers | 254ms | 0.060 | Cost-sensitive learning |
| AutoFolio | AutoML pipeline | 24ms | 0.071 | Automated configuration |
| SMAC | Bayesian optimization | 30ms | 0.066 | Random forest surrogate |
| Hyperband | Bandit allocation | 0.03ms | 0.101 | Successive halving |
| **Librex.Meta** | **Tournament + Elo** | **0.15ms** | **0.055** | **Swiss system meta-learning** |

Librex.Meta achieves best regret with near-instant selection, enabled by pre-computed Elo ratings and linear-time clustering.

---

**Word count**: ~600 words (~1 page)
**Status**: Compressed from 900 words
**Compression**: -300 words (33% reduction)
