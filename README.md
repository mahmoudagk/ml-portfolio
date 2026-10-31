# Machine Learning Portfolio

Five end-to-end tabular machine-learning projects built with Python and scikit-learn. Each project downloads a public UCI dataset, documents the cleaning steps, trains a model, and saves reproducible metrics and model artifacts.

## Projects

| Project | Task | Main model | Focus |
|---|---|---|---|
| [Bank Marketing](projects/bank-marketing/README.md) | Binary classification | Tuned histogram gradient boosting | 45K rows, 7 models, threshold analysis, campaign-duration leakage control |
| [Adult Income](projects/adult-income/README.md) | Binary classification | Balanced logistic regression | Missing-value cleanup, mixed feature types, and evaluation limits |
| [Wine Quality](projects/wine-quality/README.md) | Regression | Random forest regressor | Numeric data validation and regression metrics |
| [Covertype Advanced](projects/covertype-advanced/README.md) | 7-class classification | Tuned multi-model benchmark | 581K rows, 54 source features, feature engineering, class imbalance, CV tuning |
| [Bike Demand Forecasting](projects/bike-demand-forecast/README.md) | Hourly demand regression | Random forest benchmark winner | Chronological split, time-series tuning, leakage controls |

## Held-out results

| Project | Results |
|---|---|
| Bank Marketing | ROC-AUC **0.791**, average precision **0.451**, F1 **0.489** (validation-selected threshold) |
| Adult Income | ROC-AUC **0.904**, F1 **0.672** (official UCI test split) |
| Wine Quality | MAE **0.467**, R² **0.460** (mean baseline MAE: 0.705) |
| Covertype Advanced | Random Forest; test macro-F1 **0.874**, balanced accuracy **0.886** |
| Bike Demand Forecasting | Random Forest; test MAE **56.55 bikes/hour**, R² **0.850** (chronological holdout) |

See each project's metrics JSON for the full metric set and split details. These are educational baselines, not production claims.

## Run everything

Requirements: Python 3.10 or newer.

```bash
python -m pip install -r requirements.txt
python run_all.py
```

The scripts download the source data from UCI when run. They write cleaned CSV files to each project's `data/processed/` folder and save a fitted model plus `metrics.json` in `artifacts/`. Raw downloads are not committed.

To regenerate the result images for the original three projects, run `python scripts/plot_results.py`. The standalone forest-cover and bike-demand projects write their main diagnostic charts when trained; the bike project also has a separate script for its supplementary charts.

The advanced forest-cover benchmark runs separately because it trains and compares several models: `python projects/covertype-advanced/src/train.py`. The bike-demand forecast also runs separately: `python projects/bike-demand-forecast/src/train.py`.

## Data and attribution

All five datasets are from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/). See each project README for its dataset citation and task-specific notes. UCI lists these datasets under CC BY 4.0; attribution is included in the project documentation. The code in this repository is licensed under MIT; see [LICENSE](LICENSE).

## بالعربي

المستودع فيه ٥ مشاريع: شغّل `python run_all.py` لأول ٣ مشاريع، ومشروعي الغابات وتوقع الطلب على الدراجات لكل واحد أمر تشغيل منفصل. مشروع الدراجات بيقارن ٧ نماذج بتقسيم زمني يمنع تسرب بيانات المستقبل.
