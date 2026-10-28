# Adult Income: Census Income Classification

## Goal

Classify whether annual income is above $50,000 using the UCI Adult census dataset. This historical dataset is for practicing data preparation and model evaluation; it is not appropriate for real decisions about people.

## Data cleaning and modeling

- Download UCI's `adult.data` and `adult.test` files from the official archive.
- Strip whitespace from text columns, convert `?` placeholders to missing values, and normalize the test labels (which end in a period).
- Preserve the source's train/test split. Fit numeric median imputation/scaling and categorical most-frequent imputation/one-hot encoding on the training data only.
- Fit class-weighted logistic regression and report precision, recall, F1, balanced accuracy, and ROC-AUC on the provided test split.

Run from the repository root with `python projects/adult-income/src/train.py`.

## Results

On UCI's provided test split, the model reached ROC-AUC **0.904**, recall **0.841**, and F1 **0.672**. Full results are in [`artifacts/metrics.json`](artifacts/metrics.json).

## Data source

Becker, B. & Kohavi, R. (1996). [Adult, UCI Machine Learning Repository](https://doi.org/10.24432/C5XW20). Dataset license: CC BY 4.0.

## Limitations

The records come from 1994 U.S. census data, and the target reflects historical income inequality. Accuracy or other aggregate scores do not measure whether errors are distributed fairly across demographic groups. This model is educational only and should not be used for employment, lending, eligibility, or other consequential decisions.
