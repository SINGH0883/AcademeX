# AcademeX: Predictive Analytics for Student Performance in Exams
**Course:** Data Science Fundamentals and Statistical Intuition  
**Target:** L&T EduTech Project-1 Submission  
**Repository:** [https://github.com/SINGH0883/AcademeX](https://github.com/SINGH0883/AcademeX)  

---

## 📌 Project Overview
**AcademeX** is an end-to-end machine learning and statistical intuition project developed to forecast student examination scores ($G3$) using demographic, social, and academic behavioral indicators. The project leverages Multiple Linear Regression alongside regularized models (Ridge and Lasso) and translates regression outputs into academic classification metrics (Confusion Matrix, Precision, Recall, F1-Score) based on standard passing criteria ($G3 \ge 10/20$).

---

## 📅 7-Day Project Plan

For full details, see [`plan.md`](plan.md).

- **Day 1 (Completed ✅):** Data Collection, Preprocessing, Encoding & Feature Engineering (`prior_avg_grade`, `support_index`, `passed`).
- **Day 2 (Completed ✅):** Exploratory Data Analysis (EDA) & Statistical Intuition.
  - **Part 1/4 (Completed ✅):** Univariate Descriptive Statistics & Feature Profiling (`src/eda_descriptive_stats.py`, `reports/`).
  - **Part 2/4 (Completed ✅):** Target Variable Distribution, Q-Q Plot & Normality Testing (`src/eda_target_normality.py`, `figures/`).
  - **Part 3/4 (Completed ✅):** Correlation & Multicollinearity Analysis (`src/eda_correlations.py`, `figures/correlation_heatmap.png`, `reports/`).
  - **Part 4/4 (Completed ✅):** Feature Deep Dives & Longitudinal Visualizations (`src/eda_deep_dives.py`, `figures/`).
- **Day 3 (In Progress 🔄):** Baseline Regression & Theoretical Foundations.
  - **Part 1/4 (Completed ✅):** OLS Mathematical Formulation & Univariate Baselines (`src/baseline_univariate_models.py`, `figures/`).
  - **Part 2/4 (Completed ✅):** Academic Core Baseline & Multivariate Progression (`src/baseline_core_model.py`, `figures/`).
  - **Part 3/4 (Pending ⏳):** Gauss-Markov Assumptions Verification & Residual Diagnostics.
  - **Part 4/4 (Pending ⏳):** Multicollinearity Diagnostics & Variance Inflation Factor (VIF).
- **Day 4 (Pending ⏳):** Multiple Linear Regression & Regularization (Ridge / Lasso).
- **Day 5 (Pending ⏳):** Model Diagnostics, Evaluation & Confusion Matrix.
- **Day 6 (Pending ⏳):** Interactive Prediction & Academic Advisory System Deployment.
- **Day 7 (Pending ⏳):** Final Report Compilation & Submission-ready PDF Generation.

---

## 📂 Repository Structure

```text
├── plan.md                          # Comprehensive 7-day working plan
├── README.md                        # Project documentation
├── .gitignore                       # Git ignore file
├── data/
│   ├── student-mat.csv              # Raw UCI Math student dataset
│   ├── student-por.csv              # Raw UCI Portuguese student dataset
│   └── student_clean.csv            # Cleaned, encoded & engineered dataset
├── src/
│   ├── preprocess.py                # Data loading, cleaning, encoding & splitting
│   ├── eda.py                       # Statistical visualizations and EDA scripts
│   ├── model.py                     # Linear Regression training, evaluation & diagnostics
│   └── deployment.py                # Interactive CLI prediction and recommendation system
├── figures/                         # High-resolution charts (300 DPI)
└── models/
    └── linear_regression_model.joblib # Serialized model bundle
```

---

## 🚀 Quickstart & Usage

### 1. Requirements
Ensure Python 3.10+ is installed along with dependencies:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn reportlab joblib
```

### 2. Preprocess Data
```bash
python src/preprocess.py
```

### 3. Run Exploratory Data Analysis (EDA)
```bash
python src/eda.py
```

### 4. Train & Evaluate Models
```bash
python src/model.py
```

### 5. Run Interactive Prediction Deployment
```bash
python src/deployment.py
```

---

## 📊 Key Results Benchmark
- **Linear Regression $R^2$:** ~0.77 - 0.79
- **Root Mean Squared Error (RMSE):** ~2.04 - 2.14 grade points
- **Classification Accuracy ($G3 \ge 10$):** ~81.0%
- **F1-Score:** ~0.835
