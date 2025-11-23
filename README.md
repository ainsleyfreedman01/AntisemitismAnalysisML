# Antisemitism Trends Analysis

This is the analysis repository for campus antisemitism incidents and related trends.  
It centralizes the data cleaning, holiday-labeling, visualization, and modeling used to explore patterns of incidents and test hypotheses such as whether protests are more likely on Jewish holidays.

## Project Motivation

The raw incident records and prior analyses were fragmented and not easily reproducible.

This project was created to:
- Centralize and clean campus incident data for reproducible analysis  
- Provide a transparent, versioned pipeline for labeling and testing hypotheses (e.g., holiday associations)  
- Produce clear visualizations and inspection artifacts for reviewers and stakeholders  
- Make the code and results accessible to researchers and recruiters without relying on opaque spreadsheets or ad-hoc scripts

The focus wasn’t just analysis — it was creating a sustainable, well-documented pipeline that others can extend.

## What Went Into Building It

This project emphasized data integrity, reproducibility, and clear presentation of results.

Major areas of focus included:
- Cleaning and normalizing raw incident rows from `data/campus_reports.csv`  
- Implementing robust holiday labeling (including a canonical per-day expansion of user-provided Jewish holiday ranges)  
- Aggregating events into daily metrics and building inspection CSVs for manual review  
- Building short, executable notebooks to reproduce figures and statistical tests  
- Structuring outputs and models so results are easy to share (`outputs/`, `models/`)  
- Designing the code so it runs portably (parquet→CSV fallback, venv-aware instructions)

A lot of effort went into balancing thorough data validation with usability for non-technical reviewers.

## Tech Stack

- **Language**: Python 3.11 (virtualenv: `antisemitism_env`)  
- **Data & Analysis**: `pandas`, `numpy`, `scipy` (Fisher exact), `scikit-learn`  
- **NLP / Utilities**: `nltk`  
- **Visualization**: `matplotlib`, `seaborn`, `plotly`  
- **Notebooks**: Jupyter / nbconvert  
- **Persistence**: lightweight `pickle` for models, CSV/parquet for tabular artifacts  
- **Version Control**: Git & GitHub

These choices prioritize reproducibility, scientific tooling, and portability across environments.

## Development Process

This was an iterative, data-first process focused on reproducible results.

Some key elements of the workflow included:
- Iteratively cleaning and validating raw inputs and edge cases in `scripts/`  
- Expanding user-supplied holiday ranges into canonical per-day holiday records for robust labeling  
- Creating short notebooks (`notebooks/experiments_more_likely_on_holidays.updated.ipynb`) that load processed metrics and display matched raw events  
- Adding IO fallbacks so notebooks run in minimal environments (try parquet, fall back to CSV)  
- Running statistical tests and saving inspection artifacts (`outputs/protest_jewish_holiday_metrics_corrected.csv`, `outputs/inspection_jh_raw_matches.csv`)  
- Packaging model artefacts into `models/` and keeping notebooks and outputs organized for reviewers

The process emphasized small, verifiable steps and clear artifacts that others can inspect.

## My Contributions

I specifically worked on:
- Implementing the data cleaning and aggregation scripts in `scripts/analysis_scripts/` and `scripts/machine_learning_scripts/`  
- Building the `hardcoded_jewish_holidays.py` helper to expand holiday ranges and integrate it as a reliable fallback  
- Fixing the pipeline bugs (datetime coercion for `pd.Grouper`, parquet→CSV fallbacks) that blocked reproducible notebook runs  
- Creating and executing `notebooks/experiments_more_likely_on_holidays.updated.ipynb` and producing an executed copy for reviewers  
- Producing inspection CSVs and plots in `outputs/` and moving trained model pickles into `models/` for tidy packaging  
- Writing this README and cleaning transient files so the repo is presentable to recruiters and collaborators

This work strengthened skills in data engineering for research, reproducible analysis, and communicating results through notebooks and artifacts.