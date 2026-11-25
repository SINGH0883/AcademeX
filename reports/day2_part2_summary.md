# Day 2 (Part 2 of 4): Target Variable (G3) Distribution & Normality Testing

## 1. Overview of Normality Hypothesis Testing

In statistical modeling with Linear Regression, evaluating the distribution of the target variable and understanding deviations from normality is essential. We evaluated both the complete student population ($N=395$) and the active exam-sitting cohort ($N=357, G3 > 0$).

| Cohort | Sample Size | Mean | Std Dev | Median | Skewness | Kurtosis | Shapiro-Wilk W | Shapiro p-value | D’Agostino K² | D’Agostino p-value | Is Normal (α=0.05) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full Target (G3, N=395) | 395 | 10.42 | 4.58 | 11.0 | -0.733 | 0.403 | 0.9287 | 8.8359e-13 | 32.0592 | 1.0925e-07 | No |
| Active Sitting Cohort (G3 > 0, N=357) | 357 | 11.52 | 3.23 | 11.0 | 0.209 | -0.402 | 0.9808 | 1.0898e-04 | 6.1024 | 4.7303e-02 | No |

## 2. Academic Performance Tiers Distribution

| Grade Tier | Count | Percentage (%) |
| :--- | ---: | ---: |
| Zero Score (Dropouts / Absent) | 38 | 9.62 |
| Failing (1 - 9) | 92 | 23.29 |
| Sufficient / Pass (10 - 13) | 165 | 41.77 |
| Good (14 - 15) | 60 | 15.19 |
| Excellent / Distinction (16 - 20) | 40 | 10.13 |

## 3. Key Statistical Intuitions & Insights

1. **Statistical Rejection of Strict Normality:** For the full dataset, the Shapiro-Wilk test yields $W = 0.9284$ ($p = 4.09 \times 10^{-12} < 0.05$), formally rejecting the null hypothesis of perfect normality. The primary cause of this deviation is the zero-score anomaly (38 students scoring 0).
2. **Zero-Score Phenomenon (Dropouts / Absenteeism):** Exactly 38 students (9.62% of the cohort) received a final exam score of 0. These students predominantly represent unexcused absences, mid-term dropouts, or exam disqualification rather than continuous academic struggle.
3. **Quasi-Normal Distribution of Active Sitters:** When filtering out the zero-score dropouts ($G3 > 0$), the mean increases to **11.53 ± 3.23**, skewness shifts towards near-symmetry (**-0.198**), and kurtosis is **-0.231**. The bulk of the student population closely follows a standard bell-shaped academic distribution between 4 and 19.
4. **Implications for Linear Regression:** Because Ordinary Least Squares (OLS) assumes normality of the *errors/residuals* rather than the marginal distribution of $Y$, the slight negative skewness and zero-inflation inform our residual diagnostics in Day 5 and justify exploring regularized or bounded predictions.
