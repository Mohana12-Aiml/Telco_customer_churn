# Customer Churn Prediction and Segmentation using Machine Learning

A machine learning pipeline that predicts telecom customer churn with a tuned Decision Tree classifier and segments the customer base into behavioural groups with K-Means clustering. Built as an AI & ML internship capstone project under the PEARL Program (Q2D / IBM Innovation Centre for Education).

## Overview

Telecom providers lose a measurable share of subscribers every month, and manual monitoring can't flag at-risk customers before they leave. This project builds an end-to-end pipeline — from a raw customer workbook to business recommendations — that does two things at once:

- **Predicts** which active customers are likely to churn, and explains *why*, using an interpretable Decision Tree.
- **Segments** the customer base into four behavioural groups with K-Means, independently of the churn label, to show which kinds of customers are driving the risk.

The two methods converge on the same conclusion: **tenure, monthly charges, and contract type** are the dominant drivers of churn.

## Results

| Metric | Score |
|---|---|
| Accuracy | 0.744 |
| Precision (Churn) | 0.602 |
| Recall (Churn) | 0.716 |
| F1-Score (Churn) | 0.654 |
| ROC AUC | 0.816 |

- Tuned via 5-fold cross-validated grid search (`criterion=entropy`, `max_depth=4`, `min_samples_leaf=50`) with `class_weight='balanced'`, selected on F1 rather than accuracy to avoid rewarding a majority-class predictor.
- Three features — `MonthlyCharges`, `tenure`, `Contract` — account for ~89% of total feature importance.
- K-Means (k=4, chosen via elbow + silhouette diagnostics) identifies segment churn rates ranging from **8% to 57%**, with the two high-churn segments separated from the two low-churn segments by exactly the same two features the tree splits on first.

## Repository Contents

```
├── telco_customer.ipynb                          # Full analysis notebook (EDA, model, clustering)
├── Telco_Customer_Churn.xlsx                      # Source dataset (2,000 records, 21 columns)
├── telco_churn_with_clusters.csv                  # Output: original data enriched with cluster labels
├── Internship_Report_Telco_Customer_Churn.docx    # Full written report
├── Telco_Customer_Churn_Presentation.pptx         # Slide deck summarising the project
└── README.md
```

## Methodology

1. **Data cleaning** — `TotalCharges` type-coerced; missing values (all zero-tenure customers) imputed with the customer's own `MonthlyCharges` rather than a column mean; target mapped to binary.
2. **EDA** — churn examined against contract type, tenure, monthly charges, and technical support.
3. **Modelling** — categorical features label-encoded; stratified 75/25 train-test split; Decision Tree tuned via `GridSearchCV` (24 combinations × 5 folds).
4. **Evaluation** — accuracy, precision, recall, F1, ROC AUC, confusion matrix, plus extracted decision rules and feature importances for interpretability.
5. **Segmentation** — 12 behavioural features scaled and clustered with K-Means (churn label excluded); segments profiled by size, tenure, spend, and churn rate.

## Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `Matplotlib` · `Seaborn`

## Getting Started

```bash
pip install pandas numpy scikit-learn matplotlib seaborn openpyxl jupyter
jupyter notebook telco_customer.ipynb
```

Run all cells top to bottom. The notebook reads `Telco_Customer_Churn.xlsx` and writes `telco_churn_with_clusters.csv` on completion.

## Key Findings

- Month-to-month customers churn at **more than twice** the rate of two-year contract customers (43.5% vs 18.2%).
- Customers without technical support churn considerably more than those who subscribe to it.
- Demographic attributes (gender, senior citizen status, dependents) carry **zero** predictive importance — churn here is driven by the commercial relationship, not who the customer is.
- The highest-risk segment (57% churn) is short-tenure, high-spend, month-to-month customers — exactly where retention spend should be concentrated first.

## Author

**Sunkara Mohana Sruthi**
B.Tech CSE (AI & ML) — Internship under the PEARL Program, Q2D (Quantum Quotient Decode) in collaboration with IBM Innovation Centre for Education (IBM ICE)

## License

This project is submitted as academic coursework. Feel free to fork and adapt for learning purposes.
