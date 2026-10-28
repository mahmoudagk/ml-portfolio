# Bank Marketing: Term Deposit Prediction

## Goal

Predict whether a customer subscribed to a term deposit during a Portuguese bank's phone campaign. This is a learning project, not a tool for deciding whom a bank should contact.

## Data cleaning and modeling

- Download the UCI ZIP archive and read `bank-full.csv` using its semicolon delimiter.
- Trim text fields, normalize the target, and drop exact duplicate rows.
- Treat UCI's `unknown` category as missing; impute categorical values inside the training pipeline so the test set does not influence preprocessing.
- Exclude `duration`: the call length is only known after a call and would leak information in a pre-call targeting scenario.
- One-hot encode categories, scale numeric features, and fit a class-weighted logistic regression.
- Use a stratified 80/20 split and report precision, recall, F1, balanced accuracy, and ROC-AUC.

Run from the repository root with `python projects/bank-marketing/src/train.py`. The cleaned data, model, and metrics are written under this project.

## Results

On a stratified 20% holdout, the model reached ROC-AUC **0.752**, recall **0.596**, and F1 **0.370**. Full results and split details are in [`artifacts/metrics.json`](artifacts/metrics.json).

## Data source

Moro, S., Rita, P., & Cortez, P. (2014). [Bank Marketing, UCI Machine Learning Repository](https://doi.org/10.24432/C5K306). Dataset license: CC BY 4.0.

## Limitations

The data records past campaign outcomes and may reflect historical selection bias. The reported scores do not establish business value or fairness. Use a time-based validation split and cost-sensitive evaluation before any operational use.
