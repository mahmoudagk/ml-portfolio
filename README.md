# Machine Learning Portfolio

Three end-to-end tabular machine-learning projects built with Python and scikit-learn. Each project downloads a public UCI dataset, documents the cleaning steps, trains a model, and saves reproducible metrics and model artifacts.

## Projects

| Project | Task | Main model | Focus |
|---|---|---|---|
| [Bank Marketing](projects/bank-marketing/README.md) | Binary classification | Balanced logistic regression | Imbalanced labels and avoiding campaign-duration leakage |
| [Adult Income](projects/adult-income/README.md) | Binary classification | Balanced logistic regression | Missing-value cleanup, mixed feature types, and evaluation limits |
| [Wine Quality](projects/wine-quality/README.md) | Regression | Random forest regressor | Numeric data validation and regression metrics |

## Held-out results

| Project | Results |
|---|---|
| Bank Marketing | ROC-AUC **0.752**, F1 **0.370** (stratified 20% test split) |
| Adult Income | ROC-AUC **0.904**, F1 **0.672** (official UCI test split) |
| Wine Quality | MAE **0.467**, R² **0.460** (mean baseline MAE: 0.705) |

See each `artifacts/metrics.json` for the full metric set and split details. These are educational baselines, not production claims.

## Run everything

Requirements: Python 3.10 or newer.

```bash
python -m pip install -r requirements.txt
python run_all.py
```

The scripts download the source data from UCI when run. They write cleaned CSV files to each project's `data/processed/` folder and save a fitted model plus `metrics.json` in `artifacts/`. Raw downloads are not committed.

## Data and attribution

All three datasets are from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/). See each project README for its dataset citation and task-specific notes. UCI lists these datasets under CC BY 4.0; attribution is included in the project documentation. The code in this repository is licensed under MIT; see [LICENSE](LICENSE).

## بالعربي

المستودع فيه ٣ مشاريع كاملة: تنظيف البيانات، تجهيز الخصائص، تدريب المودل، وحفظ المقاييس والمودل. شغّل `python run_all.py` بعد تثبيت المتطلبات. البيانات الخام تُنزّل عند التشغيل، والبيانات المنظفة والنتائج تُحفظ داخل كل مشروع.
