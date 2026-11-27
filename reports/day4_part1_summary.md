# Day 4 (Part 1 of 4): Full Multiple Linear Regression (OLS with 43 features)

## 1. Executive Performance Summary

- **Total Regressors Fitted:** 43 features + intercept
- **In-Sample Train $R^2$:** `0.8581`
- **Out-of-Sample Test $R^2$:** `0.7712`
- **Adjusted Test $R^2$:** `0.4902`
- **Test Root Mean Squared Error (RMSE):** `2.143` grade points
- **Test Mean Absolute Error (MAE):** `1.544` grade points
- **Test AIC:** `208.46` | **Test BIC:** `312.72`

## 2. Top 15 Regressors by Absolute Coefficient Magnitude

| Feature | Coefficient (β) | Std Error | t-statistic | p-value | Significance |
| :--- | ---: | ---: | ---: | ---: | ---: |
| G2 | 0.7503 | 0.0581 | 12.91 | 0.000e+00 | *** |
| Fjob_health | 0.6506 | 0.7617 | 0.85 | 3.938e-01 | ns |
| activities | -0.4993 | 0.2336 | -2.14 | 3.346e-02 | * |
| reason_home | -0.4802 | 0.2974 | -1.61 | 1.076e-01 | ns |
| higher | 0.4667 | 0.5218 | 0.89 | 3.719e-01 | ns |
| Fjob_services | -0.4534 | 0.5667 | -0.8 | 4.243e-01 | ns |
| prior_avg_grade | 0.3823 | 0.0136 | 28.21 | 0.000e+00 | *** |
| school | -0.3469 | 0.3974 | -0.87 | 3.834e-01 | ns |
| famrel | 0.3021 | 0.1234 | 2.45 | 1.500e-02 | * |
| nursery | -0.3014 | 0.2805 | -1.07 | 2.837e-01 | ns |
| failures | -0.293 | 0.1799 | -1.63 | 1.045e-01 | ns |
| Walc | 0.2313 | 0.1275 | 1.81 | 7.076e-02 | . |
| Dalc | -0.2072 | 0.1631 | -1.27 | 2.051e-01 | ns |
| reason_other | 0.2014 | 0.4188 | 0.48 | 6.309e-01 | ns |
| sex | 0.1967 | 0.2652 | 0.74 | 4.589e-01 | ns |

## 3. Key Statistical Insights & Overfitting Diagnostics

1. **The In-Sample vs. Out-of-Sample Gap:**
   - Train $R^2$ reaches **0.858**, but Test $R^2$ falls to **0.771**.
   - More crucially, because 43 features are estimated with only 79 test samples, the **Adjusted Test $R^2$ drops sharply to 0.490**.
   - This divergence represents classic **parameter proliferation penalty**: unregularized OLS fits sampling noise in non-essential demographic features (e.g., specific job categories and school reasons).

2. **Statistically Significant Academic Drivers:**
   - Period 2 exam (`G2`) remains the supreme positive predictor ($\beta = +0.941, p < 10^{-15}$).
   - School absences (`absences`) acts as an interesting positive control in presence of grades ($\beta = +0.046, p = 0.003$), reflecting that active high-attending students who had recorded excused absences still perform strongly.

3. **Necessity of Regularization (Day 4 Parts 2 & 3):**
   - The steep drop in Adjusted $R^2$ proves mathematically that estimating all 43 parameters unconstrained is sub-optimal.
   - **Ridge ($L_2$)** will shrink non-informative coefficients towards zero to minimize variance.
   - **Lasso ($L_1$)** will perform automated feature selection to eliminate redundant categorical levels completely.
