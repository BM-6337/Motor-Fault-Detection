# Motor Fault Detection from Industrial Sensor Data

Predicting machine failure before it happens, using motor sensor readings — the same underlying idea behind condition-based maintenance systems used to avoid costly unplanned downtime.

## Goal

Train a classification model on motor sensor data (air/process temperature, rotational speed, torque, tool wear) to predict whether a machine is healthy or about to fail.

## Dataset

[AI4I 2020 Predictive Maintenance Dataset](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification) (Kaggle) — 10,000 records of synthetic but realistically-distributed industrial sensor data.

| Column | Description |
|---|---|
| `Type` | Product quality grade (L / M / H) |
| `Air temperature [K]` | Ambient air temperature |
| `Process temperature [K]` | Process temperature |
| `Rotational speed [rpm]` | Motor rotational speed |
| `Torque [Nm]` | Torque applied |
| `Tool wear [min]` | Tool wear time |
| `Target` | Binary label — 0 = healthy, 1 = failure |
| `Failure Type` | Failure category (only populated when `Target` = 1) |

## Pipeline

1. **Load & inspect** — confirm the data is actually clean (no missing values) rather than assuming it
2. **EDA** — class balance, boxplots of each sensor reading split by outcome, correlation heatmap
3. **Feature engineering** — two physics-motivated features (mechanical power, temperature differential), label-encode `Type`, then stratified 80/20 train/test split
4. **Baseline model** — Logistic Regression (simple, fast, fully explainable)
5. **Main model** — Random Forest (usually higher accuracy on tabular sensor data, still interpretable via feature importances)
6. **Evaluation** — accuracy, precision, recall, F1, and confusion matrices for both models
7. **Feature importance** — which sensor reading drives the failure prediction most
8. **(Optional) SHAP** — per-prediction explainability layered on top of the global feature importances

## Why accuracy alone isn't enough here

Failures are rare — only ~3% of records. A model that always predicts "healthy" would already score ~96-97% accuracy while being useless. Precision, recall, and F1 are tracked alongside accuracy throughout, and `class_weight="balanced"` is used during training so the rare failure class doesn't get ignored.

## Results

Random Forest outperformed the Logistic Regression baseline on every tracked metric (accuracy, precision, recall, F1) and was used for the final feature-importance analysis.

## Tools

- **Data:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Modeling:** scikit-learn (`LogisticRegression`, `RandomForestClassifier`, `train_test_split`, `classification_report`, `confusion_matrix`)
- **Explainability (optional):** SHAP

## Next steps

- Address class imbalance with SMOTE/oversampling instead of class weighting alone
- Try a gradient-boosted model (XGBoost / LightGBM)
- Swap in real vibration-spectrum data (e.g. MAFAULDA) for a more physically-grounded fault signature

## License

MIT
