# Day 2 (Part 3 of 4): Correlation & Multicollinearity Analysis

## 1. Overview
Correlation analysis quantifies the strength, direction, and statistical significance of linear (Pearson) and monotonic (Spearman) relationships between independent predictors and final exam achievement ($G3$). Additionally, multicollinearity screening identifies inter-predictor dependencies that impact Multiple Linear Regression modeling.

## 2. Ranked Predictors by Correlation with Final Grade (G3)

| Feature | Pearson r | Pearson p-value | Spearman rho | Spearman p-value | Significance | Direction |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| G2 | 0.905 | 7.626e-148 | 0.957 | 1.425e-213 | *** (p < 0.001) | Positive (+) |
| prior_avg_grade | 0.89 | 5.824e-136 | 0.944 | 1.618e-191 | *** (p < 0.001) | Positive (+) |
| G1 | 0.801 | 9.001e-90 | 0.878 | 8.141e-128 | *** (p < 0.001) | Positive (+) |
| failures | -0.36 | 1.466e-13 | -0.361 | 1.282e-13 | *** (p < 0.001) | Negative (-) |
| Medu | 0.217 | 1.336e-05 | 0.225 | 6.290e-06 | *** (p < 0.001) | Positive (+) |
| higher | 0.182 | 2.668e-04 | 0.172 | 5.781e-04 | *** (p < 0.001) | Positive (+) |
| Fedu | 0.152 | 2.380e-03 | 0.17 | 6.896e-04 | ** (p < 0.01) | Positive (+) |
| goout | -0.133 | 8.229e-03 | -0.166 | 9.193e-04 | ** (p < 0.01) | Negative (-) |
| romantic | -0.13 | 9.713e-03 | -0.091 | 6.940e-02 | ** (p < 0.01) | Negative (-) |
| traveltime | -0.117 | 1.987e-02 | -0.121 | 1.655e-02 | * (p < 0.05) | Negative (-) |
| internet | 0.098 | 5.048e-02 | 0.108 | 3.217e-02 | ns (Not Sig) | Positive (+) |
| studytime | 0.098 | 5.206e-02 | 0.105 | 3.667e-02 | ns (Not Sig) | Positive (+) |
| health | -0.061 | 2.239e-01 | -0.048 | 3.435e-01 | ns (Not Sig) | Negative (-) |
| Dalc | -0.055 | 2.785e-01 | -0.121 | 1.618e-02 | ns (Not Sig) | Negative (-) |
| Walc | -0.052 | 3.032e-01 | -0.104 | 3.797e-02 | ns (Not Sig) | Negative (-) |
| famrel | 0.051 | 3.086e-01 | 0.055 | 2.757e-01 | ns (Not Sig) | Positive (+) |
| absences | 0.034 | 4.973e-01 | 0.018 | 7.254e-01 | ns (Not Sig) | Positive (+) |
| freetime | 0.011 | 8.227e-01 | -0.005 | 9.212e-01 | ns (Not Sig) | Positive (+) |
| support_index | 0.005 | 9.282e-01 | -0.043 | 3.904e-01 | ns (Not Sig) | Positive (+) |

## 3. Multicollinearity Screening (Inter-Predictor Associations)

| Feature 1 | Feature 2 | Pearson r | p-value | Collinearity Level |
| :--- | ---: | ---: | ---: | ---: |
| G2 | G1 | 0.852 | 1.441e-112 | High (|r| >= 0.7) |
| Dalc | Walc | 0.648 | 2.548e-48 | Moderate (|r| >= 0.5) |
| Medu | Fedu | 0.623 | 6.419e-44 | Moderate (|r| >= 0.5) |
| goout | Walc | 0.42 | 2.402e-18 | Mild (|r| >= 0.3) |
| G2 | failures | -0.356 | 3.078e-13 | Mild (|r| >= 0.3) |
| G1 | failures | -0.355 | 3.727e-13 | Mild (|r| >= 0.3) |
| failures | higher | -0.3 | 1.123e-09 | Mild (|r| >= 0.3) |

## 4. Key Statistical Intuitions & Architectural Takeaways

1. **Decisive Primacy of Prior Academic Performance:** Period 2 grade ($G2, r = +0.905, p = 1.06 \times 10^{-148}$) and Period 1 grade ($G1, r = +0.801, p = 2.08 \times 10^{-89}$) exhibit overwhelmingly strong, statistically significant positive correlations with final exam outcomes.
2. **Strongest Academic Inversion Factor:** Past class failures (`failures`, $r = -0.360, p = 2.05 \times 10^{-13}$) is the single strongest negative predictor of exam performance in the entire dataset.
3. **Positive Socio-Academic Catalysts:**
   - Mother's education (`Medu`, $r = +0.217, p = 1.34 \times 10^{-5}$) and Father's education (`Fedu`, $r = +0.152, p = 2.45 \times 10^{-3}$).
   - Higher education aspirations (`higher`, $r = +0.182, p = 2.76 \times 10^{-4}$).
   - Weekly study time (`studytime`, $r = +0.098, p = 0.051$).
4. **Significant Multicollinearity Detected:**
   - $G1$ and $G2$ share severe collinearity ($r = 0.852, p < 10^{-100}$). This explains why combining both in Ordinary Least Squares requires careful feature formulation or regularization (Ridge/Lasso) to prevent coefficient instability.
   - Parental education ($Medu$ and $Fedu$) is moderately collinear ($r = 0.623$).
   - Alcohol consumption ($Dalc$ and $Walc$) is moderately collinear ($r = 0.648$).
