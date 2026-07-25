# Employee Attrition Prediction (HR Analytics)

## Overview
This project analyzes an HR dataset to understand why employees leave a company
and builds a predictive model to identify employees at risk of attrition.
The analysis includes data cleaning, exploratory data analysis (EDA),
statistical testing, and machine learning model comparison.

## Dataset
- Source: [Kaggle - HR Analytics and Job Prediction](https://www.kaggle.com/datasets/mfaisalqureshi/hr-analytics-and-job-prediction?resource=download)
- ~15,000 employee records with features including satisfaction level,
  evaluation scores, number of projects, monthly hours, tenure, and more.

## Key Steps
1. **Data Cleaning** — checked for missing values, identified and removed ~3,000
   duplicate rows using a probability-based justification.
2. **Outlier Detection** — used the IQR method to flag potential outliers
   (e.g., in tenure).
3. **EDA** — visualized relationships between satisfaction, hours worked,
   tenure, and attrition; identified overworked and underworked employee
   clusters among those who left.
4. **Statistical Testing** — used the Box-Tidwell test to check the linearity
   assumption required for logistic regression; found it violated for most
   features, motivating a tree-based modeling approach.
5. **Modeling** — trained and tuned Decision Tree and Random Forest classifiers
   using GridSearchCV, evaluated with accuracy, precision, recall, F1, and ROC AUC.
6. **Feature Engineering** — removed `satisfaction_level` (a self-reported,
   not-always-reliable feature) and engineered a binary `overworked` feature
   based on monthly hours, to build a more deployable model.

## Results
- Final Random Forest model achieved an AUC of ~93.8%+ on the test set,
  with strong precision, recall, and accuracy.
- Random Forest modestly outperformed the Decision Tree model.

## Key Findings & Recommendations
- Employees are frequently overworked, and this is strongly linked to attrition.
- Long-tenured employees (4+ years) show unexpectedly low satisfaction 
  worth investigating or addressing via promotion.
- Recommendations include capping project load, fairly compensating overtime,
  and adjusting evaluation criteria to avoid rewarding excessive hours alone.

## Tools Used
Python, pandas, NumPy, scikit-learn, XGBoost, statsmodels, matplotlib, seaborn

## How to Run
This notebook was originally developed on Kaggle. To run locally:
1. Clone this repo
2. Install dependencies: `pip install -r requirements.txt`
3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mfaisalqureshi/hr-analytics-and-job-prediction?resource=download) and place it in the working directory
4. Open `notebook.ipynb` in Jupyter or VS Code
