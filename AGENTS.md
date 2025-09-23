# Repository Guidelines

## Project Structure & Module Organization
`cli.py` is the operational gateway and delegates most work to `src/`, where `hologram_analysis.py` drives feature extraction and training, `anomaly_detector.py` encapsulates Mahalanobis scoring, and `dataset_unifier.py` keeps datasets consistent. Raw holograms live in `holograms/`, intermediate CSV or parquet assets in `data/`, and generated models, plots, and reports in `results/`. Shared configuration sits in `config.yaml` with supporting presets inside `config/`, while manuscripts and write-ups stay under `manuscrito/`.

## Build, Test, and Development Commands
Set up the environment via `python -m pip install -r requirements.txt`. Execute a full pipeline with `python cli.py analyze`, toggling turnaround with `--mode quick` or `--mode deep`. Validate anomaly behaviour using `python cli.py validate --image-dir ./holograms`, and lean on `python example_single_prediction.py` for rapid API smoke-checks.

## Coding Style & Naming Conventions
Use four-space indentation, snake_case for functions and variables, and PascalCase for classes (`HologramAnalyzer`, `DistanceBasedAnomalyDetector`). Retain the Spanish messaging and emoji tone already present in CLI prints. Add type hints when touching function signatures and keep constants uppercase. Format Python changes with `black` (line length 88) and organize imports with `isort` before committing.

## Testing Guidelines
House automated checks in a `tests/` package (create it if absent) and exercise workflows with `pytest -q`. Start with smoke tests that call `python cli.py analyze --mode quick` against sample data, then cover edge cases such as missing config keys or absent image directories when modifying detectors or preprocessing helpers. Document any large test datasets in the PR so reviewers can reproduce failures locally.

## Commit & Pull Request Guidelines
Git history shows terse imperative subjects (`fix`, `outliers y otras cositas`); continue that style but add a few extra words of context. Keep commits scoped to one concern and avoid mixing data drops with code. Pull requests should outline motivation, key results, and reproduction commands, link related issues or experiment logs, and attach screenshots or metrics whenever outputs change.

## Configuration & Data Handling
Keep adjustable settings in `config.yaml` or sibling templates instead of hardcoding paths. Large models or proprietary holograms belong in `results/` or `holograms/` and should stay out of version control—share download links when necessary. Update `.gitignore` entries if new artifact folders are introduced.
