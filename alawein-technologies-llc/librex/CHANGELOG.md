# Changelog

All notable changes to Librex will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-11-08

### Added

#### Core Framework

- Universal problem adapter architecture
- Centralized method registry with automatic discovery

#### Problem Adapters (initial set)

- QAP (Quadratic Assignment Problem)
- TSP (Traveling Salesman Problem)
- Portfolio (mean–variance)

#### Baseline Methods (initial set)

- random_search (QAP, TSP)
- greedy_qap, local_search_qap (QAP)
- two_opt_tsp (TSP)
- portfolio_equal_weights, portfolio_pgd (Portfolio)

#### Infrastructure

- Tests scaffold with coverage config
- Pre-commit hooks for code quality
- CI pipelines for tests, lint, and docs

#### Documentation

- High-level docs and migration plans
- Examples for QAP, TSP, Portfolio

## [Unreleased]

### Planned

- Migrate advanced methods from reference archive
- Add benchmark runners and publish results
- Expand domain adapters (VRP, scheduling) as scoped
- Documentation site (Sphinx/RTD) build workflow

---

## Version History

- **1.0.0** (2025-11-08): Initial public release of adapter framework and baseline methods

---

For detailed changes, see the [commit history](https://github.com/yourusername/Librex/commits/main).
