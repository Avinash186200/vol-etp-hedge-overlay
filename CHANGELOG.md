# Changelog

All notable changes to this repository are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.2.0] - 2026-11-15

### Changed
- Docs: training-grade README and walkthrough. The README now has a plain-English problem statement, a pipeline
  diagram, key concepts, "What you will learn", a clearer synthetic label and an updated layout. New
  `docs/walkthrough.md`.
- README quick start: the `--prices my.csv` placeholder is replaced by commands that run as written.

### Added
- `scripts/make_synthetic_data.py`, which writes synthetic closes to `data/prices.csv`, with a test.

## [1.1.0] - 2026-11-15

### Added
- `volhedge.diagnostics` and `scripts/leakage_diagnostics.py`: label-overlap checks on scores only (IC by lead and
  the refit-cycle fingerprint), with tests on planted cases, and a README section on what a later review confirmed.
- `original/`: the original research code, sanitised (names, credentials, data and notebook outputs removed).
- `CHANGELOG.md`, `CONTRIBUTING.md` and a documentation index in `docs/README.md`.

## [1.0.0] - 2026-11-05

### Added
- First public release: the maintained code, documentation, tests and CI.
