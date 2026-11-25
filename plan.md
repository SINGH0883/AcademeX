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
| **Day 2 (1/4)** | **Univariate Descriptive Statistics & Profiling** | • Compute parametric & non-parametric central tendency (mean, median, mode).<br>• Evaluate dispersion (variance, std dev, IQR, range, CV%).<br>• Distribution asymmetry (skewness & kurtosis).<br>• Stratified subgroup breakdowns (study hours vs grades, failures vs grades). | **COMPLETED** ✅ | • `src/eda_descriptive_stats.py`<br>• `reports/day2_part1_descriptive_stats.csv`<br>• `reports/day2_part1_studytime_breakdown.csv`<br>• `reports/day2_part1_summary.md` |
| **Day 2 (2/4)** | **Target Variable Distribution & Normality** | • Shapiro-Wilk & D'Agostino-Pearson normality tests.<br>• Density distribution & zero-grade dropout analysis.<br>• Academic grade category distribution. | **COMPLETED** ✅ | • `src/eda_target_normality.py`<br>• `figures/target_distribution_and_qq.png`<br>• `figures/grade_tiers_breakdown.png`<br>• `reports/day2_part2_normality_tests.csv`<br>• `reports/day2_part2_grade_tiers.csv`<br>• `reports/day2_part2_summary.md` |
| **Day 2 (3/4)** | **Correlation & Association Analysis** | • Pearson & Spearman rank correlation matrices.<br>• Heatmap visualization and statistical significance testing.<br>• Multicollinearity screening between predictors. | **COMPLETED** ✅ | • `src/eda_correlations.py`<br>• `figures/correlation_heatmap.png`<br>• `figures/top_correlations_bar.png`<br>• `reports/day2_part3_correlation_matrix.csv`<br>• `reports/day2_part3_feature_correlations_ranked.csv`<br>• `reports/day2_part3_summary.md` |
| **Day 2 (4/4)** | **Feature Deep Dives & Visualization** | • Study time vs final grades boxplots & violin plots.<br>• Prior grades ($G1, G2$) linear regression fit scatter.<br>• School absences & class failures risk interaction.<br>• Parental education & aspirations interaction. | **COMPLETED** ✅ | • `src/eda_deep_dives.py`<br>• `figures/studytime_vs_grade.png`<br>• `figures/prior_grades_vs_final.png`<br>• `figures/absences_failures_effect.png`<br>• `figures/parental_edu_and_aspirations.png`<br>• `reports/day2_part4_deep_dive_metrics.csv`<br>• `reports/day2_part4_summary.md` |
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

### Day 2 (Part 1 of 4): Completed
- Built dedicated analytical module: [`src/eda_descriptive_stats.py`](src/eda_descriptive_stats.py).
- Calculated comprehensive parametric & non-parametric summary statistics (Mean, Median, Std Dev, Variance, IQR, Min, Max, Skewness, Kurtosis, CV%) across all numeric academic features.
- Generated subgroup cross-tabulations analyzing final grades ($G3$) across Weekly Study Time categories and Past Class Failures.
- Exported audit artifacts:
  - [`reports/day2_part1_descriptive_stats.csv`](reports/day2_part1_descriptive_stats.csv)
  - [`reports/day2_part1_studytime_breakdown.csv`](reports/day2_part1_studytime_breakdown.csv)
  - [`reports/day2_part1_summary.md`](reports/day2_part1_summary.md)

### Day 2 (Part 2 of 4): Completed
- Built target evaluation module: [`src/eda_target_normality.py`](src/eda_target_normality.py).
- Conducted hypothesis tests for normality: Shapiro-Wilk ($W=0.9287, p=8.84\times 10^{-13}$), D'Agostino-Pearson omnibus test ($K^2=32.06, p=1.09\times 10^{-7}$).
- Isolated and analyzed zero-score phenomenon: 38 students (9.62%) with $G3=0$ representing unexcused exam absenteeism/dropouts. Evaluated active sitting cohort ($N=357, G3>0$) showing quasi-normal distribution ($\mu=11.52, \text{Skew}=0.209$).
- Classified student outcomes into 5 academic grade tiers: Distinction (10.13%), Good (15.19%), Pass (41.77%), Failing (23.29%), Zero-score dropouts (9.62%).
- Generated diagnostic visualizations:
  - [`figures/target_distribution_and_qq.png`](figures/target_distribution_and_qq.png) (Empirical KDE, theoretical normal PDF fit, Normal Q-Q plot)
  - [`figures/grade_tiers_breakdown.png`](figures/grade_tiers_breakdown.png) (Academic tier count and percentage distribution)
- Exported audit artifacts:
  - [`reports/day2_part2_normality_tests.csv`](reports/day2_part2_normality_tests.csv)
  - [`reports/day2_part2_grade_tiers.csv`](reports/day2_part2_grade_tiers.csv)
  - [`reports/day2_part2_summary.md`](reports/day2_part2_summary.md)

### Day 2 (Part 3 of 4): Completed
- Built correlation engine: [`src/eda_correlations.py`](src/eda_correlations.py).
- Computed Pearson linear ($r$) and Spearman monotonic ($\rho$) correlation matrices with exact two-tailed $p$-values.
- Ranked all candidate predictors: Identified top positive drivers ($G2: r=+0.905, G1: r=+0.801, Medu: r=+0.217, higher: r=+0.182$) and strongest negative penalties ($failures: r=-0.360, goout: r=-0.133, Dalc: r=-0.055$).
- Screened for multicollinearity: Detected severe inter-predictor collinearity between $G1$ and $G2$ ($r = 0.852, p < 10^{-100}$), and moderate collinearity between parental education ($Medu-Fedu, r = 0.623$) and alcohol consumption ($Dalc-Walc, r = 0.648$).
- Generated publication-grade diagnostic plots:
  - [`figures/correlation_heatmap.png`](figures/correlation_heatmap.png) (Annotated lower-triangle correlation heatmap)
  - [`figures/top_correlations_bar.png`](figures/top_correlations_bar.png) (Ranked bar chart of top positive and negative predictors)
- Exported audit artifacts:
  - [`reports/day2_part3_correlation_matrix.csv`](reports/day2_part3_correlation_matrix.csv)
  - [`reports/day2_part3_feature_correlations_ranked.csv`](reports/day2_part3_feature_correlations_ranked.csv)
  - [`reports/day2_part3_summary.md`](reports/day2_part3_summary.md)

### Day 2 (Part 4 of 4): Completed
- Built comprehensive visual deep-dive module: [`src/eda_deep_dives.py`](src/eda_deep_dives.py).
- Bivariate Regression Trajectories: Fitted and plotted $G1 \rightarrow G3$ ($R^2 = 0.642, \beta_1 = 1.106$) and $G2 \rightarrow G3$ ($R^2 = 0.819, \beta_1 = 1.102$) demonstrating that Period 2 exams drive over 80% of final grade variability.
- Quantified failure penalty cliff: Mean scores drop sharply from 11.25 (0 failures) to 8.12 (1 failure) and 5.69 (3 failures).
- Analyzed parental education & aspiration interaction: Higher education aspirations (`higher`) serve as a resilient protective buffer, boosting grades across all parental education tiers.
- Generated high-resolution visualization suite (300 DPI):
  - [`figures/studytime_vs_grade.png`](figures/studytime_vs_grade.png)
  - [`figures/prior_grades_vs_final.png`](figures/prior_grades_vs_final.png)
  - [`figures/absences_failures_effect.png`](figures/absences_failures_effect.png)
  - [`figures/parental_edu_and_aspirations.png`](figures/parental_edu_and_aspirations.png)
- Exported audit artifacts:
  - [`reports/day2_part4_deep_dive_metrics.csv`](reports/day2_part4_deep_dive_metrics.csv)
  - [`reports/day2_part4_summary.md`](reports/day2_part4_summary.md)
