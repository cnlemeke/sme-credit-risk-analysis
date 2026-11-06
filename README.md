# SME Financial Risk and Business Intelligence Analysis

An end-to-end analysis of 350 synthetic data of small and medium enterprises (SMEs). It covers data cleaning, exploration, feature engineering, a modelling test and a rule-based risk scorecard.

The headline result is honest rather than flattering: with this data, non-financial features do not predict which firms make a loss. The notebook shows how I tested that and why a perfect-looking score would have been a mistake.

**Key Findings**

- **Cleaning.** 350 raw records became 328 analysis records. The issue log in section 2 lists 14 problems, each with a row count. 5 near-duplicates were removed and 17 records outside the SME definition went to a separate file.
- **Losses and leverage.** 114 of 328 SMEs (34.8%) made a loss. 60 firms (18.3%) have debt-to-equity above 2.
- **Risk tiers.** A two-signal scorecard puts 178 firms in Green, 126 in Amber and 24 in Red.
- **Prediction.** Models built on non-profit features did not predict losses. The best mean ROC-AUC was 0.551, against 0.500 for a model with no information.
- **Group differences.** The gaps I tested by sector, company size and digital adoption were not distinguishable from chance.
- **A warning.** Adding one profit-based column lifts AUC from 0.513 to 0.995. That jump is leakage, not skill.

## Repository Structure

| Path | What it is |
|------|------------|
| `notebooks/code.ipynb` | The full analysis, saved with its outputs |
| `reports/key_figures.json` | Every number quoted in the written findings, saved by the notebook |
| `data/sme_dataset.csv` | Raw synthetic data, 350 rows and 30 columns |
| `data/sme_cleaned.csv` | The 328 analysis records after cleaning |
| `data/sme_risk_scored.csv` | Cleaned data with engineered features and the stress tier |
| `figures/` | The ten charts as PNG files |
| `src/generate_sme_data.py` | The script that builds the raw data and plants the defects |

## Method

1. **Clean.** Remove near-duplicates, standardise years, categories and dates, recalculate margin and debt-to-equity from their source columns, set blank loan amounts to zero for firms with no loan, move records outside the SME definition to a separate file, and fill the remaining gaps with flagged medians. Every step is counted in an issue log.
2. **Explore.** Sector, size, leverage, loans and digital adoption. Where a chart suggests a gap, I test it before reporting it.
3. **Build features.** Financial ratios, log versions of skewed columns, encodings and a stress score.
4. **Model.** Predict whether a firm made a loss from non-profit features. Repeated cross-validation, three models and a no-information baseline. A second check shows what happens when a profit-based column leaks in.
5. **Score.** A transparent Green, Amber, Red scorecard built from two signals, a loss and debt-to-equity above 2.

## About the data

The data is synthetic. `src/generate_sme_data.py` builds it with a fixed seed and plants defects on purpose: missing values, outliers, mixed formats, 5 near-duplicates and invalid rows. The cleaning work is real. The business patterns are not, because the generator draws most fields independently, and I say so wherever it affects a conclusion.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/generate_sme_data.py  # optional, rebuilds data/sme_dataset.csv
cd notebooks
python -m jupyter nbconvert --to notebook --execute --code.ipynb
```

Tested on Python 3.13.17 with pandas 3.0.5, numpy 2.5.3, scikit-learn 1.9.1, scipy 1.18.1, matplotlib 3.11.2, seaborn 0.13.2, nbformat 5.11.1, nbconvert 7.17.2, ipykernel 7.4.0.

## Limits

The notebook lists them in section 7.3. The main ones: the data is synthetic, it is a single snapshot, the debt-to-equity limit of 2 is my choice, and with 328 firms the tests detect only fairly large effects.


