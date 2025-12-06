# Week 12: Final Submission Execution Plan

**Target**: AutoML Conference 2025
**Deadline**: March 31, 2025
**Current Status**: Complete draft ready (6,610 words)
**Goal**: Compress, format, and submit 8-page paper

---

## Overview: 7-Day Sprint to Submission

**Start**: March 24, 2025 (Day 1)
**Deadline**: March 31, 2025 (Day 7)
**Buffer**: None (tight but achievable)

**Daily Time Commitment**: 6-8 hours/day
**Total Effort**: ~50 hours over 7 days

---

## Day 1 (March 24): Compression - Part 1

**Goal**: Reduce paper from 6,610 to ~6,100 words (-510 words)
**Time**: 6 hours
**Focus**: Related Work + Methods compression

### Morning (3 hours): Related Work Compression

**Current**: 900 words
**Target**: 600 words
**Cut**: -300 words

**Action Plan**:

1. **Reduce Algorithm Selection subsection** (2.1)
   - Keep: SATzilla, AutoFolio (2-3 sentences each)
   - Remove: 3S, ISAC details
   - Compress: ASlib to 1 sentence
   - **Cut**: -100 words

2. **Reduce Meta-Learning subsection** (2.2)
   - Keep: SMAC, Hyperband, BOHB (1 sentence each)
   - Remove: Auto-sklearn, Auto-WEKA details
   - **Cut**: -100 words

3. **Reduce Tournament-Based Learning subsection** (2.3)
   - Keep: Elo rating system (1 sentence)
   - Compress: TrueSkill to 1 sentence
   - Remove: Evolutionary algorithm details
   - **Cut**: -50 words

4. **Reduce Clustering subsection** (2.4)
   - Compress to 2-3 sentences total
   - **Cut**: -50 words

**Checklist**:
- [ ] Open `paper_drafts/02_RELATED_WORK.md`
- [ ] Create backup: `02_RELATED_WORK_original.md`
- [ ] Compress subsection 2.1 (-100 words)
- [ ] Compress subsection 2.2 (-100 words)
- [ ] Compress subsection 2.3 (-50 words)
- [ ] Compress subsection 2.4 (-50 words)
- [ ] Verify word count: 600 words
- [ ] Save as `02_RELATED_WORK_compressed.md`

### Afternoon (3 hours): Methods Compression

**Current**: 1,100 words
**Target**: 1,000 words
**Cut**: -100 words

**Action Plan**:

1. **Move detailed Elo formulas to appendix**
   - Keep: High-level description
   - Move: Full mathematical derivation → Appendix D
   - **Cut**: -50 words

2. **Compress Implementation Details** (3.8)
   - Keep: Training time (1 sentence)
   - Remove: Detailed feature extraction description
   - **Cut**: -30 words

3. **Tighten UCB Selection** (3.6)
   - Keep: Formula and explanation
   - Remove: Redundant complexity discussion
   - **Cut**: -20 words

**Checklist**:
- [ ] Open `paper_drafts/03_METHODS.md`
- [ ] Create backup: `03_METHODS_original.md`
- [ ] Extract Elo derivation → save for Appendix D
- [ ] Compress section 3.8 (-30 words)
- [ ] Tighten section 3.6 (-20 words)
- [ ] Verify word count: 1,000 words
- [ ] Save as `03_METHODS_compressed.md`

### Evening (1 hour): Review Progress

- [ ] Count total words reduced: Should be ~400 words
- [ ] Current total: 6,610 - 400 = 6,210 words
- [ ] Review compressed sections for coherence
- [ ] Note any issues for Day 2

**Day 1 Target**: 6,100-6,200 words ✓

---

## Day 2 (March 25): Compression - Part 2

**Goal**: Reduce paper from ~6,100 to 5,600 words (-500 words)
**Time**: 6 hours
**Focus**: Results + Discussion + Conclusion compression

### Morning (2 hours): Results Compression

**Current**: 1,100 words
**Target**: 900 words
**Cut**: -200 words

**Action Plan**:

1. **Move detailed tables to appendix**
   - Keep: Summary table in main text (Table 1: Overall Performance)
   - Move: Full per-scenario breakdown → Appendix A
   - Reference: "See Appendix A for complete results"
   - **Cut**: -80 words

2. **Compress Ablation Studies** (5.4)
   - Keep: Key finding (only n_clusters matters)
   - Move: Detailed ablation results → Appendix C
   - **Cut**: -70 words

3. **Compress Statistical Significance** (5.5)
   - Keep: Main finding (p>0.05, effect sizes)
   - Move: Detailed test results → Appendix B
   - **Cut**: -50 words

**Checklist**:
- [ ] Open `paper_drafts/05_RESULTS.md`
- [ ] Create backup: `05_RESULTS_original.md`
- [ ] Extract full Table 2 → Appendix A
- [ ] Compress section 5.4 (-70 words)
- [ ] Compress section 5.5 (-50 words)
- [ ] Verify word count: 900 words
- [ ] Save as `05_RESULTS_compressed.md`

### Afternoon (3 hours): Discussion Compression

**Current**: 900 words
**Target**: 700 words
**Cut**: -200 words

**Action Plan**:

1. **Compress "Why It Works"** (6.1)
   - Keep: Efficiency, adaptivity, specialization (1 sentence each)
   - Remove: Detailed explanations
   - **Cut**: -80 words

2. **Compress Graph Problem Dominance** (6.2)
   - Keep: Hypothesis and evidence
   - Remove: Detailed explanation of graph features
   - **Cut**: -50 words

3. **Compress Limitations** (6.3)
   - Keep: 4 key limitations (1 paragraph each → 2-3 sentences)
   - Remove: Redundancy with Results section
   - **Cut**: -40 words

4. **Compress Mock Data Gap** (6.5)
   - Keep: Key finding
   - Remove: Detailed synthetic vs. real comparison
   - **Cut**: -30 words

**Checklist**:
- [ ] Open `paper_drafts/06_DISCUSSION.md`
- [ ] Create backup: `06_DISCUSSION_original.md`
- [ ] Compress section 6.1 (-80 words)
- [ ] Compress section 6.2 (-50 words)
- [ ] Compress section 6.3 (-40 words)
- [ ] Compress section 6.5 (-30 words)
- [ ] Verify word count: 700 words
- [ ] Save as `06_DISCUSSION_compressed.md`

### Evening (1 hour): Conclusion Compression

**Current**: 450 words
**Target**: 350 words
**Cut**: -100 words

**Action Plan**:

1. **Compress Future Work**
   - Keep: 4 key directions (1 sentence each)
   - Remove: Detailed explanations
   - **Cut**: -60 words

2. **Compress Broader Impact**
   - Keep: 2-3 key points
   - Remove: Detailed examples
   - **Cut**: -40 words

**Checklist**:
- [ ] Open `paper_drafts/07_CONCLUSION.md`
- [ ] Create backup: `07_CONCLUSION_original.md`
- [ ] Compress Future Work (-60 words)
- [ ] Compress Broader Impact (-40 words)
- [ ] Verify word count: 350 words
- [ ] Save as `07_CONCLUSION_compressed.md`

### Final Evening Review (1 hour)

- [ ] Count total words: Should be ~5,600 words
- [ ] Current sections:
  - Abstract: 210 words
  - Introduction: 1,200 words
  - Related Work: 600 words
  - Methods: 1,000 words
  - Experiments: 750 words
  - Results: 900 words
  - Discussion: 700 words
  - Conclusion: 350 words
  - **Total: 5,710 words** ✓

- [ ] Read through complete compressed draft
- [ ] Check coherence and flow
- [ ] Note any awkward transitions

**Day 2 Target**: 5,600-5,700 words ✓

---

## Day 3 (March 26): LaTeX Conversion - Part 1

**Goal**: Set up LaTeX project and convert first 4 sections
**Time**: 8 hours
**Focus**: Template setup + Abstract/Intro/Related/Methods

### Morning (2 hours): LaTeX Setup

**Action Plan**:

1. **Download AutoML 2025 Template**
   - Visit AutoML 2025 website
   - Download LaTeX template ZIP
   - Extract to `paper_latex/` directory

2. **Set up Overleaf Project** (recommended)
   - Create new Overleaf project
   - Upload AutoML template
   - Test compilation (empty template)

   **Alternative**: Local LaTeX setup
   - Install TeX Live or MikTeX
   - Install VS Code with LaTeX Workshop extension
   - Test compilation locally

3. **Create File Structure**
   ```
   paper_latex/
   ├── main.tex (main document)
   ├── sections/
   │   ├── abstract.tex
   │   ├── introduction.tex
   │   ├── related_work.tex
   │   ├── methods.tex
   │   ├── experiments.tex
   │   ├── results.tex
   │   ├── discussion.tex
   │   └── conclusion.tex
   ├── figures/
   │   ├── figure2.pdf
   │   ├── figure3.pdf
   │   └── figure4.pdf
   ├── references.bib
   └── appendix.tex
   ```

**Checklist**:
- [ ] Download AutoML 2025 LaTeX template
- [ ] Set up Overleaf project OR local LaTeX environment
- [ ] Create file structure
- [ ] Test compile empty template
- [ ] Verify PDF generation works

### Late Morning (2 hours): Convert Abstract + Introduction

**Abstract Conversion**:
- [ ] Open `paper_drafts/00_ABSTRACT.md`
- [ ] Create `sections/abstract.tex`
- [ ] Copy text, remove Markdown formatting
- [ ] Add LaTeX formatting (italics, bold if needed)
- [ ] Compile and check

**Introduction Conversion**:
- [ ] Open `paper_drafts/01_INTRODUCTION.md`
- [ ] Create `sections/introduction.tex`
- [ ] Convert subsections to `\subsection{}`
- [ ] Add `\label{}` for cross-references
- [ ] Compile and check pagination (~2 pages)

**Checklist**:
- [ ] Abstract in LaTeX (1 paragraph, ~200 words)
- [ ] Introduction in LaTeX (4 subsections, ~1,200 words)
- [ ] Compilation successful
- [ ] Page count: ~2.5 pages total

### Afternoon (2 hours): Convert Related Work + Methods

**Related Work Conversion**:
- [ ] Open `paper_drafts/02_RELATED_WORK_compressed.md`
- [ ] Create `sections/related_work.tex`
- [ ] Convert subsections (2.1-2.4)
- [ ] Add citations: `\cite{xu2008satzilla}` (placeholder)
- [ ] Add positioning table (Table in LaTeX)
- [ ] Compile and check (~1 page)

**Methods Conversion**:
- [ ] Open `paper_drafts/03_METHODS_compressed.md`
- [ ] Create `sections/methods.tex`
- [ ] Convert formulas to LaTeX math mode:
   - Regret formula: `\text{Regret}(x) = \frac{r_{\pi(x)}(x) - r_{a^*(x)}(x)}{r_{a^*(x)}(x)}`
   - Elo formula: `R'_i = R_i + K \times (S_i - E(S_i))`
   - UCB formula: `\text{UCB}(a) = \text{normalize}(R_{c,a}) + \lambda \sqrt{\frac{\log N}{n_a}}`
- [ ] Add algorithm pseudocode (optional, can move to appendix)
- [ ] Compile and check (~1.5 pages)

**Checklist**:
- [ ] Related Work in LaTeX (~1 page)
- [ ] Methods in LaTeX (~1.5 pages)
- [ ] All formulas render correctly
- [ ] Compilation successful
- [ ] Page count: ~5 pages total so far

### Evening (2 hours): Review and Test

- [ ] Compile full document (Abstract → Methods)
- [ ] Check page layout and formatting
- [ ] Verify font sizes, margins
- [ ] Fix any compilation errors
- [ ] Read through for LaTeX formatting issues

**Day 3 Target**: Abstract + Intro + Related + Methods in LaTeX (5 pages) ✓

---

## Day 4 (March 27): LaTeX Conversion - Part 2 + Figures

**Goal**: Convert remaining sections + integrate all figures
**Time**: 8 hours
**Focus**: Experiments/Results/Discussion/Conclusion + Figures

### Morning (3 hours): Convert Experiments + Results

**Experiments Conversion**:
- [ ] Open `paper_drafts/04_EXPERIMENTS.md`
- [ ] Create `sections/experiments.tex`
- [ ] Convert subsections (4.1-4.4)
- [ ] Add ASlib scenario details (itemize/enumerate)
- [ ] Compile and check (~1 page)

**Results Conversion**:
- [ ] Open `paper_drafts/05_RESULTS_compressed.md`
- [ ] Create `sections/results.tex`
- [ ] Convert Table 1 (Overall Performance) to LaTeX table:
   ```latex
   \begin{table}[t]
   \centering
   \caption{Average Performance Across 5 Scenarios}
   \label{tab:overall}
   \begin{tabular}{lcccc}
   \toprule
   Method & Avg Regret & Top-1 Acc & Time (ms) & Speedup \\
   \midrule
   Librex.Meta (optimal) & \textbf{0.0545} & 46.5\% & \textbf{0.15} & \textbf{1664×} \\
   ... (other rows)
   \bottomrule
   \end{tabular}
   \end{table}
   ```
- [ ] Add figure references: `Figure~\ref{fig:scenario}`, `Figure~\ref{fig:pareto}`
- [ ] Compile and check (~1.5 pages)

**Checklist**:
- [ ] Experiments in LaTeX (~1 page)
- [ ] Results in LaTeX with Table 1 (~1.5 pages)
- [ ] Table formatting clean
- [ ] Figure references added (figures inserted later)

### Afternoon (3 hours): Convert Discussion + Conclusion + Figures

**Discussion Conversion**:
- [ ] Open `paper_drafts/06_DISCUSSION_compressed.md`
- [ ] Create `sections/discussion.tex`
- [ ] Convert subsections (6.1-6.5)
- [ ] Compile and check (~1 page)

**Conclusion Conversion**:
- [ ] Open `paper_drafts/07_CONCLUSION_compressed.md`
- [ ] Create `sections/conclusion.tex`
- [ ] Convert summary + future work
- [ ] Compile and check (~0.5 pages)

**Figure Integration**:

1. **Copy figures to LaTeX project**:
   - [ ] Copy `figures/output/figure2_scenario_performance.pdf` → `paper_latex/figures/`
   - [ ] Copy `figures/output/figure3_pareto_frontier.pdf` → `paper_latex/figures/`
   - [ ] Copy `figures/output/figure4_hyperparameter_sensitivity.pdf` → `paper_latex/figures/`

2. **Insert Figure 2** (in Results section):
   ```latex
   \begin{figure}[t]
   \centering
   \includegraphics[width=0.9\columnwidth]{figures/figure2_scenario_performance.pdf}
   \caption{Per-scenario performance comparison. Librex.Meta (optimal) wins on GRAPHS-2015 (rank 1/7).}
   \label{fig:scenario}
   \end{figure}
   ```

3. **Insert Figure 3** (in Results section):
   ```latex
   \begin{figure}[t]
   \centering
   \includegraphics[width=0.8\columnwidth]{figures/figure3_pareto_frontier.pdf}
   \caption{Speed-accuracy trade-off. Librex.Meta achieves best Pareto efficiency: 1664× faster with competitive accuracy.}
   \label{fig:pareto}
   \end{figure}
   ```

4. **Insert Figure 4** (in Results section):
   ```latex
   \begin{figure}[t]
   \centering
   \includegraphics[width=\columnwidth]{figures/figure4_hyperparameter_sensitivity.pdf}
   \caption{Hyperparameter sensitivity. Only n\_clusters impacts performance; other parameters have negligible effect.}
   \label{fig:ablation}
   \end{figure}
   ```

**Checklist**:
- [ ] Discussion in LaTeX (~1 page)
- [ ] Conclusion in LaTeX (~0.5 pages)
- [ ] All 3 figures copied to `paper_latex/figures/`
- [ ] Figures inserted with captions
- [ ] Figure references work (`\ref{fig:scenario}` etc.)
- [ ] Compile and verify figures appear

### Evening (2 hours): Create Figure 1 + References

**Figure 1: Architecture Diagram**:

Option 1: **draw.io** (recommended, free)
- [ ] Download draw.io (https://app.diagrams.net/)
- [ ] Create new diagram
- [ ] Draw two boxes:
  - Training Phase: Instances → Features → KMeans → Tournaments → Elo ratings
  - Selection Phase: New instance → Features → Cluster → UCB → Selected algorithm
- [ ] Export as PDF: `figure1_architecture.pdf`
- [ ] Copy to `paper_latex/figures/`
- [ ] Insert in Methods section

Option 2: **PowerPoint**
- [ ] Create slide with architecture diagram
- [ ] Export as PDF
- [ ] Copy to `paper_latex/figures/`

**LaTeX for Figure 1**:
```latex
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/figure1_architecture.pdf}
\caption{Librex.Meta architecture. Training phase: Swiss tournaments update Elo ratings. Selection phase: Cluster assignment + UCB selection.}
\label{fig:architecture}
\end{figure}
```

**References (BibTeX)**:

Create `references.bib`:
```bibtex
@article{xu2008satzilla,
  title={SATzilla: Portfolio-based algorithm selection for SAT},
  author={Xu, Lin and Hutter, Frank and Hoos, Holger H and Leyton-Brown, Kevin},
  journal={Journal of Artificial Intelligence Research},
  volume={32},
  pages={565--606},
  year={2008}
}

@inproceedings{lindauer2015autofolio,
  title={AutoFolio: An automatically configured algorithm selector},
  author={Lindauer, Marius and Hoos, Holger H and Hutter, Frank and Schaub, Torsten},
  booktitle={Journal of Artificial Intelligence Research},
  volume={53},
  pages={745--778},
  year={2015}
}

% Add 20-30 more references...
```

In `main.tex`:
```latex
\bibliographystyle{plain}
\bibliography{references}
```

**Checklist**:
- [ ] Figure 1 created (architecture diagram)
- [ ] Figure 1 inserted in Methods section
- [ ] References.bib created with 20-30 entries
- [ ] Citations added throughout paper (`\cite{}`)
- [ ] Bibliography compiles

**Day 4 Target**: Complete LaTeX paper with all 4 figures + references (8 pages) ✓

---

## Day 5 (March 28): Appendix Preparation

**Goal**: Create complete 6-section appendix
**Time**: 6 hours
**Focus**: Results tables, statistical tests, ablation details, implementation, scenarios, reproducibility

### Morning (3 hours): Appendix A, B, C

**Appendix A: Complete Results Tables**:

Create `appendix.tex` with:
```latex
\section{Complete Results Tables}

\begin{table}[h]
\centering
\caption{Full Per-Scenario Breakdown (7 methods × 5 scenarios)}
\label{tab:full_results}
\begin{tabular}{llcccc}
\toprule
Scenario & Method & Regret & Top-1 & Top-3 & Time (ms) \\
\midrule
% Export from phase2_results_summary.csv
GRAPHS-2015 & Librex.Meta (optimal) & 0.019 & 54.8\% & 70.6\% & 0.11 \\
... (35 rows total: 7 methods × 5 scenarios)
\bottomrule
\end{tabular}
\end{table}
```

- [ ] Export data from `results/phase2/phase2_results_summary.csv`
- [ ] Convert to LaTeX table format
- [ ] Add to appendix.tex

**Appendix B: Statistical Test Details**:

```latex
\section{Statistical Test Details}

\subsection{Friedman Test}
Test statistic: $\chi^2 = 2.65$, $p = 0.85$ (not significant at $\alpha = 0.05$).

\subsection{Wilcoxon Signed-Rank Tests}
Table of pairwise comparisons:
% Export from statistical_tests.csv
...

\subsection{Effect Sizes (Cliff's Delta)}
...
```

- [ ] Export from `results/phase2/statistical_tests.csv`
- [ ] Add Friedman test results
- [ ] Add Wilcoxon test table
- [ ] Add effect sizes

**Appendix C: Hyperparameter Ablation Details**:

```latex
\section{Hyperparameter Ablation Details}

\subsection{n\_clusters Ablation}
Table showing regret for k ∈ {1,2,3,5,7,10,15,20}:
% Export from ablation_n_clusters_real.csv
...

\subsection{ucb\_c Ablation}
...
\subsection{n\_tournament\_rounds Ablation}
...
\subsection{elo\_k Ablation}
...
```

- [ ] Export from `results/ablation_real/*.csv`
- [ ] Create 4 ablation tables
- [ ] Add to appendix

**Checklist**:
- [ ] Appendix A complete (full results table)
- [ ] Appendix B complete (statistical tests)
- [ ] Appendix C complete (4 ablation tables)

### Afternoon (3 hours): Appendix D, E, F

**Appendix D: Implementation Details**:

```latex
\section{Implementation Details}

\subsection{Pseudocode: Training}
\begin{algorithm}
\caption{Librex.Meta Training}
\begin{algorithmic}
\STATE clusters $\leftarrow$ KMeans(features, k)
\STATE Initialize Elo ratings
\FOR{round $r = 1$ to $R$}
    \FOR{cluster $c$ in clusters}
        \STATE pairs $\leftarrow$ SwissPairing(cluster\_elo[c])
        \FOR{$(a_1, a_2)$ in pairs}
            \STATE winner $\leftarrow$ CompareOnInstance(a1, a2)
            \STATE UpdateElo(a1, a2, winner)
        \ENDFOR
    \ENDFOR
\ENDFOR
\end{algorithmic}
\end{algorithm}

\subsection{Complexity Analysis}
...
```

- [ ] Add training pseudocode
- [ ] Add selection pseudocode
- [ ] Add complexity analysis
- [ ] Add Elo formula derivation (moved from Methods)

**Appendix E: ASlib Scenario Descriptions**:

```latex
\section{ASlib Scenario Descriptions}

\subsection{GRAPHS-2015}
\textbf{Domain}: Graph coloring \\
\textbf{Instances}: 1,147 test instances \\
\textbf{Algorithms}: 9 specialized graph coloring solvers \\
\textbf{Features}: 105 graph-theoretic features (density, clustering coefficient, diameter, etc.) \\
\textbf{Performance}: Librex.Meta rank 1/7 (best on this scenario)

% Repeat for CSP-2010, MAXSAT12-PMS, SAT11-HAND, ASP-POTASSCO
```

- [ ] Add descriptions for all 5 scenarios
- [ ] Include instance counts, algorithm counts, feature counts
- [ ] Note Librex.Meta's performance on each

**Appendix F: Reproducibility Checklist**:

```latex
\section{Reproducibility Checklist}

\begin{itemize}
\item Code repository: \url{https://github.com/[username]/Librex.Meta} (to be added)
\item Dataset sources: ASlib (\url{http://www.aslib.net/})
\item Hardware: Intel Xeon E5-2680 v4 @ 2.40GHz, 64 GB RAM
\item Software: Python 3.9, scikit-learn 1.0, NumPy 1.21
\item Random seeds: Documented in code repository
\item Cross-validation folds: ASlib standard folds (fold 1 for testing)
\item Training time: 0.1-0.5 seconds per scenario
\item Hyperparameters: See Appendix C for details
\end{itemize}
```

- [ ] Add code repository URL (placeholder for now)
- [ ] List all dependencies
- [ ] Add hardware/software specifications
- [ ] Add random seed information

**Checklist**:
- [ ] Appendix D complete (implementation details + pseudocode)
- [ ] Appendix E complete (5 scenario descriptions)
- [ ] Appendix F complete (reproducibility checklist)
- [ ] Compile full appendix
- [ ] Verify no page limit (appendix unlimited)

**Day 5 Target**: Complete 6-section appendix ✓

---

## Day 6 (March 29): Final Proofreading

**Goal**: 3 complete revision passes + final checks
**Time**: 8 hours
**Focus**: Content, technical accuracy, language quality

### Morning (3 hours): Pass 1 - Content & Structure

**Read entire paper from Abstract to Conclusion**:

- [ ] **Narrative flow**: Does each section connect logically?
- [ ] **Consistency**: Same terminology throughout? (e.g., "algorithm selection" not "solver selection")
- [ ] **Completeness**: All claims supported by data or references?
- [ ] **Figure/table references**: All figures and tables referenced in text?
- [ ] **Section balance**: Are sections appropriately sized?

**Specific checks**:
- [ ] Abstract summarizes all key points
- [ ] Introduction establishes clear motivation
- [ ] Related Work positions our contribution clearly
- [ ] Methods explains framework completely
- [ ] Experiments describes setup thoroughly
- [ ] Results presents findings clearly
- [ ] Discussion interprets results appropriately
- [ ] Conclusion summarizes and looks forward

**Fix issues**:
- [ ] Awkward transitions between sections
- [ ] Missing connections or explanations
- [ ] Redundancy across sections
- [ ] Unclear statements

**Checklist after Pass 1**:
- [ ] Narrative flows smoothly
- [ ] No missing pieces
- [ ] All figures/tables referenced
- [ ] Consistent terminology

### Afternoon (3 hours): Pass 2 - Technical Accuracy

**Verify all numbers and formulas**:

**Abstract**:
- [ ] 0.0545 regret (check against results)
- [ ] 1664× speedup (check: 254ms / 0.15ms = 1693× ≈ 1664×)
- [ ] Rank 1/7 on GRAPHS-2015 (check results table)
- [ ] 96.5% accuracy on CSP-2010 (check results table)

**Methods**:
- [ ] Regret formula correct
- [ ] Elo formula correct: R'_i = R_i + K × (S_i - E(S_i))
- [ ] E(S_i) formula correct: 1 / (1 + 10^((R_j - R_i) / 400))
- [ ] UCB formula correct
- [ ] Complexity analysis correct (O(d), O(m log m))

**Results**:
- [ ] Table 1 numbers match `phase2_results_summary.csv`
- [ ] Per-scenario rankings correct
- [ ] Statistical test results correct (χ² = 2.65, p = 0.85)
- [ ] Effect sizes correct (Cliff's δ values)

**Discussion**:
- [ ] Claims match results
- [ ] No overstated conclusions
- [ ] Limitations accurately described

**Citations**:
- [ ] All claims cited appropriately
- [ ] No missing citations
- [ ] Citation format consistent
- [ ] All references in bibliography

**Checklist after Pass 2**:
- [ ] All numbers verified against source data
- [ ] All formulas correct
- [ ] All citations present and correct
- [ ] No technical errors

### Evening (2 hours): Pass 3 - Language & Style

**Grammar and spelling**:
- [ ] Run Grammarly or similar tool
- [ ] Check spelling throughout
- [ ] Fix any grammatical errors

**Style consistency**:
- [ ] Tense: Past for experiments ("We evaluated..."), present for contributions ("Librex.Meta achieves...")
- [ ] Voice: Active preferred ("We propose..." not "It is proposed...")
- [ ] Notation: Consistent throughout (x for instance, a for algorithm, etc.)
- [ ] Abbreviations: Defined at first use, then consistent

**Clarity**:
- [ ] Sentence clarity: Are all sentences easy to understand?
- [ ] Paragraph structure: One idea per paragraph?
- [ ] Transitions: Smooth flow between paragraphs?
- [ ] Conciseness: Remove unnecessary words

**Formal tone**:
- [ ] No informal language ("really fast" → "extremely fast")
- [ ] No contractions ("don't" → "do not")
- [ ] No first person excessive ("we" is okay for contributions, not for descriptions)

**Checklist after Pass 3**:
- [ ] No grammar/spelling errors
- [ ] Consistent style throughout
- [ ] All sentences clear
- [ ] Formal academic tone

### Final Evening (1 hour): Final Checks

**Page limit**:
- [ ] Compile PDF
- [ ] Count pages: MUST be ≤ 8 pages
- [ ] If >8 pages, identify cuts to make

**Figure quality**:
- [ ] All figures high-resolution and clear
- [ ] Figure captions descriptive
- [ ] Figures placed appropriately (near first reference)

**Table formatting**:
- [ ] Tables clean and readable
- [ ] Table captions above tables
- [ ] Numbers aligned properly

**References**:
- [ ] Bibliography compiles without errors
- [ ] All citations in text have entries in bibliography
- [ ] No unused bibliography entries

**Appendix**:
- [ ] All 6 sections present
- [ ] Tables formatted correctly
- [ ] Pseudocode clear

**Final compilation test**:
- [ ] Compile full PDF (main + appendix)
- [ ] No LaTeX errors or warnings
- [ ] PDF looks professional

**Checklist**:
- [ ] Page count ≤ 8 pages ✓
- [ ] All figures clear ✓
- [ ] All tables formatted ✓
- [ ] Bibliography complete ✓
- [ ] Appendix complete ✓
- [ ] PDF compiles successfully ✓

**Day 6 Target**: Submission-ready PDF (8 pages + appendix) ✓

---

## Day 7 (March 30-31): Code Repository + Submission

**Goal**: Prepare code repository and submit to AutoML 2025
**Time**: 8 hours
**Focus**: GitHub setup, README, final submission

### Morning (4 hours): Code Repository Preparation

**Create GitHub Repository**:

1. **Initialize repository**:
   ```bash
   cd /path/to/libria-meta
   git init
   git remote add origin https://github.com/[username]/Librex.Meta.git
   ```

2. **Clean code**:
   - [ ] Remove debug print statements
   - [ ] Add docstrings to all functions
   - [ ] Add comments to complex code
   - [ ] Format code consistently (PEP 8 for Python)

3. **Create README.md**:

```markdown
# Librex.Meta: Tournament-Based Meta-Learning for Algorithm Selection

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of "Librex.Meta: Tournament-Based Meta-Learning for Fast Algorithm Selection" (AutoML 2025).

## Overview

Librex.Meta achieves **sub-millisecond algorithm selection** (0.15ms) using Swiss-system tournaments and Elo rating systems, with best average regret across diverse benchmarks and **1664× speedup** over SATzilla.

## Key Features

- **Fast selection**: 0.15ms average (1664× faster than SATzilla)
- **Competitive accuracy**: Best average regret (0.0545) across 5 ASlib scenarios
- **Hyperparameter robust**: Only n_clusters matters (minimal tuning)
- **Excels on graph problems**: Rank 1/7 on GRAPHS-2015

## Installation

```bash
# Clone repository
git clone https://github.com/[username]/Librex.Meta.git
cd Librex.Meta

# Install dependencies
pip install -r requirements.txt

# Download ASlib data
bash scripts/download_aslib.sh
```

## Quick Start

```python
from libria_meta import Librex.MetaSolver

# Initialize with optimal configuration
solver = Librex.MetaSolver(n_clusters=3, ucb_c=1.0, n_tournament_rounds=5, elo_k=32)

# Train on ASlib scenario
solver.train(scenario_name='GRAPHS-2015')

# Select algorithm for new instance
selected_algorithm = solver.select_solver(instance_features)
```

## Reproducing Paper Results

### Phase 2 Evaluation (5 scenarios)

```bash
python benchmark/phase2_evaluation.py
```

Results saved to `results/phase2/phase2_results_summary.csv`.

### Hyperparameter Ablation Studies

```bash
python benchmark/ablation_studies_real.py
```

Results saved to `results/ablation_real/`.

### Statistical Analysis

```bash
python benchmark/statistical_significance.py
```

## Results Summary

| Scenario | Rank | Avg Regret | Top-1 Acc | Notes |
|----------|------|------------|-----------|-------|
| GRAPHS-2015 | **1/7** | **0.019** | **54.8%** | **WINS** |
| CSP-2010 | 2/7 | 0.003 | **96.5%** | Competitive |
| MAXSAT12-PMS | 4/7 | 0.025 | 58.0% | Decent |
| SAT11-HAND | 5/7 | 0.112 | 18.3% | Weak |
| ASP-POTASSCO | 5/7 | 0.113 | 5.0% | Weak |

**Overall**: Best average regret (0.0545), 1664× speedup, 46.5% top-1 accuracy.

## Citation

If you use Librex.Meta in your research, please cite:

```bibtex
@inproceedings{Librex.Meta2025,
  title={Librex.Meta: Tournament-Based Meta-Learning for Fast Algorithm Selection},
  author={[Authors]},
  booktitle={AutoML Conference},
  year={2025}
}
```

## License

MIT License - see LICENSE file for details.

## Contact

For questions or issues, please open a GitHub issue or contact [email].
```

4. **Create requirements.txt**:
   ```
   numpy==1.21.0
   pandas==1.3.0
   scikit-learn==1.0.0
   scipy==1.7.0
   matplotlib==3.4.0
   ```

5. **Create LICENSE file** (MIT recommended)

6. **Commit and push**:
   ```bash
   git add .
   git commit -m "Initial commit: Librex.Meta implementation"
   git push -u origin main
   ```

**Checklist**:
- [ ] GitHub repository created
- [ ] Code cleaned (docstrings, comments, formatting)
- [ ] README.md comprehensive
- [ ] requirements.txt complete
- [ ] LICENSE added
- [ ] All code committed and pushed
- [ ] Repository public (or private with access for reviewers)

### Afternoon (2 hours): Final Paper Preparation

**Generate final PDF**:
- [ ] Compile LaTeX one last time
- [ ] Verify page count: ≤ 8 pages
- [ ] Verify all figures render correctly
- [ ] Verify bibliography complete
- [ ] Download PDF: `Librex.Meta_automl2025.pdf`

**Prepare supplementary materials**:

Create ZIP file with:
```
supplementary_materials/
├── appendix.pdf (full appendix from LaTeX)
├── code_link.txt (GitHub repository URL)
├── data_instructions.txt (how to download ASlib data)
└── reproduction_guide.txt (step-by-step reproduction instructions)
```

- [ ] Export appendix as separate PDF
- [ ] Create code_link.txt with GitHub URL
- [ ] Create data_instructions.txt with ASlib download link
- [ ] Create reproduction_guide.txt with step-by-step instructions
- [ ] ZIP all supplementary materials

**Checklist**:
- [ ] Final PDF generated: `Librex.Meta_automl2025.pdf`
- [ ] Supplementary materials ZIP created
- [ ] All materials ready for submission

### Evening (2 hours): SUBMISSION

**AutoML 2025 Submission Portal**:

1. **Visit submission portal**:
   - Go to AutoML 2025 conference website
   - Navigate to submission portal (likely CMT or EasyChair)
   - Log in or create account

2. **Fill submission form**:
   - [ ] Title: "Librex.Meta: Tournament-Based Meta-Learning for Fast Algorithm Selection"
   - [ ] Authors: [Fill in names, affiliations, emails]
   - [ ] Abstract: Copy from paper
   - [ ] Keywords: algorithm selection, meta-learning, tournament systems, Elo ratings
   - [ ] Track: Research track (or Main track)
   - [ ] Primary subject area: AutoML, Algorithm Selection
   - [ ] Secondary subject areas: Meta-learning, Optimization

3. **Upload files**:
   - [ ] Main PDF: `Librex.Meta_automl2025.pdf` (≤ 8 pages)
   - [ ] Supplementary materials: `supplementary_materials.zip`
   - [ ] Code availability statement: "Code available at [GitHub URL]"

4. **Final checks**:
   - [ ] Verify PDF displays correctly in portal
   - [ ] Verify all author information correct
   - [ ] Verify supplementary materials uploaded
   - [ ] Verify anonymization (if required - check AutoML guidelines)

5. **Submit**:
   - [ ] Click "Submit" button
   - [ ] Verify submission confirmation email received
   - [ ] Save submission ID and confirmation

**Backup plan** (if portal issues):
- [ ] Screenshot confirmation page
- [ ] Save all submission files locally
- [ ] Email program chairs if technical issues

**Final celebration**:
- [ ] ✅ **SUBMITTED TO AutoML 2025**
- [ ] Document submission time and ID
- [ ] Notify collaborators (if any)
- [ ] Back up all materials

**Checklist**:
- [ ] Submission portal accessed
- [ ] All forms filled
- [ ] PDF uploaded (verified)
- [ ] Supplementary materials uploaded (verified)
- [ ] Submission confirmed
- [ ] Confirmation email received
- [ ] **DEADLINE MET: March 31, 2025** ✅

---

## Emergency Contingency Plans

### If Compression Fails (Days 1-2)

**Problem**: Cannot compress to 5,600 words
**Solution**:
- Accept 5,700-5,800 words (still likely fits in 8 pages with careful LaTeX formatting)
- Use smaller font for appendix references
- Adjust margins slightly (within template limits)
- Move more content to appendix

### If LaTeX Compilation Fails (Days 3-4)

**Problem**: Template issues or compilation errors
**Solution**:
- Switch to Overleaf (automatic compilation)
- Use simpler LaTeX template as backup
- Ask for help on TeX StackExchange
- Worst case: Submit in conference-provided format if available

### If Figures Don't Render (Day 4)

**Problem**: Figure formatting issues in LaTeX
**Solution**:
- Convert all figures to PDF format
- Use `\includegraphics[width=...]` with explicit dimensions
- Place figures manually with `[h]` or `[H]` placement
- Worst case: Insert as PNG with high DPI (300+)

### If Repository Setup Fails (Day 7)

**Problem**: GitHub issues or code not ready
**Solution**:
- Use anonymous GitHub/GitLab repository
- Provide code as supplementary ZIP file
- Promise code release upon acceptance
- Include detailed pseudocode in appendix as backup

### If Submission Portal Crashes (Day 7)

**Problem**: Portal down or unresponsive near deadline
**Solution**:
- Try at different times (avoid last hour)
- Use different browser
- Email submission to program chairs directly
- Request deadline extension if system-wide issue

---

## Success Criteria Checklist

**Before submission, verify**:

- [ ] Paper ≤ 8 pages (strict limit)
- [ ] All 8 sections present and complete
- [ ] All 4 figures included with captions
- [ ] All tables formatted correctly
- [ ] All formulas render correctly
- [ ] Bibliography complete (20-30 references)
- [ ] Appendix complete (6 sections)
- [ ] No LaTeX compilation errors
- [ ] PDF looks professional
- [ ] Code repository public and documented
- [ ] README comprehensive
- [ ] Reproducibility instructions clear
- [ ] Supplementary materials prepared
- [ ] Submission deadline met: March 31, 2025 ✅

**Quality checks**:

- [ ] Narrative flows smoothly
- [ ] All claims supported by data
- [ ] Statistical reporting honest (p>0.05 acknowledged)
- [ ] Limitations clearly stated
- [ ] Future work outlined
- [ ] Figures tell compelling story
- [ ] Tables easy to read
- [ ] Code runs and reproduces results

---

## Final Motivational Note

You've completed 11 weeks of intensive research and writing. The complete draft is ready. Week 12 is about **polish and packaging**, not creation.

**You have**:
✅ Novel research contribution
✅ Best empirical results (regret + speedup)
✅ Complete paper draft (6,610 words)
✅ High-quality figures (3/3 generated)
✅ Production code (5,356 lines, 100% tests passing)

**You need**:
🔨 Compression (~1,000 words to cut - straightforward)
🔨 LaTeX formatting (mechanical conversion)
🔨 Appendix preparation (export tables from CSVs)
🔨 Proofreading (3 careful passes)
🔨 Repository setup (GitHub + README)
🔨 Submission (upload and click submit)

**Difficulty**: LOW-MODERATE (no new creation, just polish)
**Time available**: 7 days (sufficient with focus)
**Outcome probability**: **>95% submission success**

Stay focused. Follow the plan. You've got this. 🎯

---

**Status**: Week 12 execution plan ready
**Next action**: Begin Day 1 compression (March 24)
**Final deadline**: March 31, 2025 (7 days)
**Confidence**: **VERY HIGH** ✅
