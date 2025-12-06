# 2. Related Work

Librex.Meta builds on three research threads: algorithm selection, meta-learning for hyperparameter optimization, and tournament-based learning systems. We position our contributions relative to each.

## 2.1 Algorithm Selection

**Classical approaches.** The algorithm selection problem was formalized by Rice (1976), who posed the question: given multiple algorithms and a problem instance, which algorithm should we run? Early work focused on hand-crafted heuristics and expert systems. Modern approaches leverage machine learning to predict algorithm performance from instance features.

**SATzilla** (Xu et al., 2008, 2012) pioneered the use of regression models for algorithm selection in SAT solving. SATzilla extracts features from CNF formulas, trains cost-sensitive regression models to predict solver runtimes, and selects the algorithm with minimum predicted cost. To reduce feature extraction overhead, SATzilla employs pre-solvers—cheap algorithms run before feature extraction to filter easy instances. While highly accurate (winning multiple SAT competitions), SATzilla's selection overhead averages 254ms, limiting applicability to real-time systems.

**AutoFolio** (Lindauer et al., 2015) extends this paradigm by automatically configuring the algorithm selection pipeline itself. Rather than manually choosing classifiers and feature subsets, AutoFolio uses automated machine learning (AutoML) to optimize the entire selection system. This yields strong performance across diverse domains but still requires 24ms selection time on average.

**3S** (Kadioglu et al., 2010) takes a schedule-based approach, running multiple solvers in sequence with time budgets allocated by learned policies. ISAC (Kadioglu et al., 2011) combines instance-specific algorithm configuration with selection. While effective, these methods introduce scheduling overhead incompatible with sub-millisecond requirements.

**ASlib** (Bischl et al., 2016) established a standardized benchmark suite for algorithm selection, providing 45+ scenarios across SAT, CSP, QBF, ASP, and other domains. ASlib enables reproducible comparison and has driven progress in the field. Our evaluation uses 5 diverse ASlib scenarios.

**Limitation.** State-of-the-art algorithm selection methods achieve strong accuracy but high latency (24-254ms). Librex.Meta addresses this gap, targeting sub-millisecond selection through tournament-based meta-learning.

## 2.2 Meta-Learning for Algorithm Selection

**Hyperparameter optimization (HPO)** methods treat algorithm selection as a black-box optimization problem: find the best algorithm (or configuration) by sequential evaluation.

**SMAC** (Hutter et al., 2011) uses sequential model-based algorithm configuration, building random forest surrogates of algorithm performance and optimizing acquisition functions to select evaluations. SMAC excels at configuration but requires 30ms+ for selection due to random forest inference.

**Hyperband** (Li et al., 2017) takes a bandit-based approach, allocating resources adaptively using successive halving. Hyperband achieves fast selection (0.03ms) by avoiding expensive models, but suffers poor accuracy on algorithm selection tasks (top-1 accuracy: 19.7% in our experiments). The issue is insufficient exploitation of problem structure.

**BOHB** (Falkner et al., 2018) combines Bayesian optimization with Hyperband, using kernel density estimators to guide resource allocation. While more accurate than Hyperband alone, BOHB still struggles on algorithm selection (19.7% top-1 accuracy), likely because bandit methods lack mechanisms to exploit problem features.

**Auto-sklearn** (Feurer et al., 2015) and Auto-WEKA (Kotthoff et al., 2017) extend meta-learning to full ML pipeline construction, learning to select preprocessing, algorithms, and hyperparameters jointly. These systems demonstrate meta-learning's power but target offline pipeline optimization, not real-time selection.

**Gap.** Existing meta-learning approaches either sacrifice speed (SMAC, Auto-sklearn) or accuracy (Hyperband, BOHB). Librex.Meta achieves both through tournament-based learning, which efficiently captures algorithm strengths while enabling fast selection via pre-computed Elo ratings.

## 2.3 Tournament-Based Learning

**Elo rating system** (Elo, 1978) was developed to rank chess players based on match outcomes. Each player has a rating (typically initialized to 1500), updated after games based on expected vs actual outcomes. Elo ratings elegantly handle partial information and converge quickly.

**TrueSkill** (Herbrich et al., 2007) extends Elo to team games using Bayesian inference, maintaining uncertainty estimates over skill levels. TrueSkill has been applied to matchmaking in online gaming and sports analytics.

**Applications in machine learning.** Tournament selection appears in evolutionary algorithms as a fitness-based selection mechanism. Active learning research has explored tournaments for query selection, asking which data points should be labeled. However, we are unaware of prior work applying Swiss-system tournaments and Elo ratings to algorithm selection meta-learning.

**Our contribution.** Librex.Meta introduces tournament-based meta-learning to algorithm selection, using Swiss-system tournaments for efficient algorithm ranking and dual Elo ratings (global + cluster-specific) for specialized performance tracking. This combination is novel.

## 2.4 Clustering for Problem Space Partitioning

**Instance space analysis** (Smith-Miles & Bowly, 2015) studies the relationship between problem features and algorithm performance, visualizing instance distributions and identifying regions of competence for different algorithms. This work motivates clustering as a mechanism for specialization.

**Feature clustering for algorithm selection** appears in several systems. Leyton-Brown et al. (2009) cluster SAT instances and train specialized models per cluster. Hutter et al. (2014) use hierarchical clustering to partition configuration spaces.

**Our approach.** Librex.Meta uses KMeans clustering on instance features to partition the problem space, then maintains cluster-specific Elo ratings to capture specialized algorithm performance. This differs from prior work in two ways: (1) we use clusters to specialize Elo ratings rather than train separate models, reducing complexity; (2) we discover that coarse clustering (k=3) outperforms fine-grained partitioning (k=20), contrary to intuition.

## 2.5 Positioning

Librex.Meta occupies a unique position in the algorithm selection landscape. Compared to classical methods (SATzilla, AutoFolio), we sacrifice ~10% accuracy for 1600x speedup. Compared to fast baselines (Hyperband), we improve accuracy dramatically (46.5% vs 19.7% top-1) with similar latency. Our tournament-based framework provides a new meta-learning approach that balances speed and accuracy effectively.

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

**Word count**: ~900 words (~1.5 pages)
**Status**: Complete draft
**Next**: Draft Methods section
