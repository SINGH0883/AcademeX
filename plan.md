# 7-Day Working Plan & Project Roadmap
## Project-1: Predictive Analytics for Student Performance in Exams
**Course:** Data Science Fundamentals and Statistical Intuition  
**Target Organization:** L&T EduTech  
**Author / Student:** Student Submission  
**Dataset:** [UCI Machine Learning Repository — Student Performance Dataset](https://archive.ics.uci.edu/dataset/320/student+performance)  

---

## 1. Project Objective
The objective of this project is to develop an end-to-end predictive analytics system to estimate and analyze student academic performance in final examinations ($G3$) based on study hours, prior academic grades ($G1, G2$), absences, past failures, and demographic/socio-economic variables using Machine Learning (Multiple Linear Regression).

The final project will be delivered as a comprehensive, publication-quality **PDF Report** and an interactive **Deployment Prediction System** for the L&T EduTech student portal.

---

## 2. 7-Day Structured Working Plan

| Day | Project Phase | Core Tasks & Sub-steps | Status | Deliverables |
| :--- | :--- | :--- | :---: | :--- |
| **Day 1** | **Data Collection & Preprocessing** | • Gather and extract UCI student performance data.<br>• Perform data hygiene, profiling, and missing value checks.<br>• Binary mapping & one-hot encoding for nominal attributes.<br>• Feature engineering (`prior_avg_grade`, `support_index`, `passed`).<br>• Save clean dataset & stratified train/test split. | **COMPLETED** ✅ | • `data/student-mat.csv`<br>• `src/preprocess.py`<br>• `data/student_clean.csv` |
| **Day 2** | **Exploratory Data Analysis (EDA) & Statistical Intuition** | • Calculate descriptive statistics (mean, median, variance, IQR).<br>• Pearson & Spearman correlation analysis.<br>• Target variable distribution & normality assessment.<br>• Visualize study time vs. exam grades.<br>• Analyze academic risk factors (absences and past failures). | **PENDING** ⏳ | • `src/eda.py`<br>• `figures/distribution_target.png`<br>• `figures/correlation_heatmap.png`<br>• `figures/studytime_vs_grade.png`<br>• `figures/prior_grades_vs_final.png`<br>• `figures/absences_failures_effect.png` |
| **Day 3** | **Baseline Regression & Theoretical Foundations** | • Mathematical formulation of Ordinary Least Squares (OLS).<br>• Verify Gauss-Markov assumptions (Linearity, Exogeneity, Homoscedasticity).<br>• Build univariate baseline (study time $\rightarrow$ G3) and bivariate baseline (G1, G2 $\rightarrow$ G3).<br>• Assess multicollinearity via Variance Inflation Factor (VIF). | **PENDING** ⏳ | • Baseline metrics table<br>• Statistical intuition analysis |
| **Day 4** | **Multiple Linear Regression & Regularization** | • Train multivariate Linear Regression model on all engineered features.<br>• Fit regularized models: Ridge ($L_2$) and Lasso ($L_1$) for comparison.<br>• Analyze feature shrinkage and slope coefficients ($\beta_j$).<br>• Evaluate regression metrics: $R^2$, Adjusted $R^2$, MSE, RMSE, MAE. | **PENDING** ⏳ | • `src/model.py`<br>• `models/linear_regression_model.joblib`<br>• `figures/feature_coefficients.png` |
| **Day 5** | **Evaluation & Classification (Confusion Matrix)** | • Residual diagnostic analysis (Residuals vs Fitted, error normality).<br>• Discretize continuous predictions at passing threshold ($G3 \ge 10/20$).<br>• Generate and analyze Confusion Matrix (TP, FP, TN, FN).<br>• Evaluate classification metrics: Accuracy, Precision, Recall, F1-Score.<br>• Educational cost analysis of False Positives vs False Negatives. | **PENDING** ⏳ | • `figures/actual_vs_predicted.png`<br>• `figures/residual_analysis.png`<br>• `figures/confusion_matrix.png` |
| **Day 6** | **Interactive Deployment System** | • Build modular prediction system (`StudentPerformancePredictor`).<br>• Implement risk classification tiers (Distinction, Pass, At-Risk).<br>• Implement automated educational recommendations engine.<br>• Provide interactive Command-Line Interface (CLI) and batch inference. | **PENDING** ⏳ | • `src/deployment.py`<br>• CLI prediction interface<br>• Deployment validation suite |
| **Day 7** | **Final Project Report & PDF Generation** | • Synthesize all methodologies, mathematics, code, and insights.<br>• Generate high-resolution figures and data tables.<br>• Compile into formal PDF document via ReportLab.<br>• Quality audit and final submission package for L&T EduTech portal. | **PENDING** ⏳ | • `Student_Performance_Report.pdf`<br>• Final project repository archive |

---

## 3. Directory Structure

```text
d:\My Pro\L&T Pro\
│
├── plan.md                          # 7-day working plan and milestone tracking
├── data/
│   ├── student-mat.csv              # Raw UCI Math student dataset
│   ├── student-por.csv              # Raw UCI Portuguese student dataset
│   └── student_clean.csv            # Cleaned, encoded & engineered dataset
│
├── src/
│   ├── preprocess.py                # Data loading, cleaning, encoding & splitting
│   ├── eda.py                       # Statistical visualizations and EDA scripts
│   ├── model.py                     # Linear Regression training, evaluation & diagnostics
│   └── deployment.py                # Interactive CLI prediction and recommendation system
│
├── figures/                         # High-resolution charts (300 DPI)
│   ├── distribution_target.png
│   ├── correlation_heatmap.png
│   ├── studytime_vs_grade.png
│   ├── prior_grades_vs_final.png
│   ├── absences_failures_effect.png
│   ├── actual_vs_predicted.png
│   ├── residual_analysis.png
│   ├── confusion_matrix.png
│   └── feature_coefficients.png
│
├── models/
│   └── linear_regression_model.joblib # Serialized model and metadata
│
└── Student_Performance_Report.pdf   # Final submission PDF for L&T EduTech Portal
```

---

## 4. Key Metrics & Benchmarks

### Regression Metrics
- **$R^2$ (Coefficient of Determination):** Proportion of variance in final grades explained by the model.
- **Adjusted $R^2$:** Penalizes non-contributing features ($1 - (1-R^2)\frac{n-1}{n-p-1}$).
- **Mean Absolute Error (MAE):** Average magnitude of prediction errors in grade points.
- **Root Mean Squared Error (RMSE):** Penalizes larger deviation errors.

### Classification Metrics (Threshold: $G3 \ge 10$)
- **Confusion Matrix:** True Positive (Pass correctly predicted), False Positive (Fail predicted as Pass), True Negative (Fail correctly predicted), False Negative (Pass predicted as Fail).
- **Precision:** $\frac{TP}{TP + FP}$
- **Recall (Sensitivity):** $\frac{TP}{TP + FN}$
- **F1-Score:** Harmonic mean of Precision and Recall ($2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$).

---

## 5. Daily Progress Log

### Day 1: Completed
- Sourced benchmark UCI dataset (`395` student records, `33` raw attributes).
- Audited missing values (0 missing). Implemented fault-tolerant imputation.
- Applied binary mapping to 13 attributes and one-hot encoding to 4 multi-class categorical attributes.
- Engineered 3 new attributes: `prior_avg_grade`, `support_index`, and `passed`.
- Saved [student_clean.csv](data/student_clean.csv) (45 total columns) and created [preprocess.py](src/preprocess.py).
