# Day 4 (Part 3 of 4): Lasso Regression (L1 Regularization) & Sparse Feature Selection

## 1. Mathematical Formulation of L1 Regularization

Lasso (Least Absolute Shrinkage and Selection Operator) penalizes the $L_1$ norm of the regression coefficients:

$$\min_\beta \left\{ \frac{1}{2n} \|y - X \beta\|_2^2 + \alpha \|\beta\|_1 \right\} = \min_\beta \left\{ \frac{1}{2n} \sum_{i=1}^n (y_i - x_i^T \beta)^2 + \alpha \sum_{j=1}^p |\beta_j| \right\}

Due to the non-differentiable sharp corners of the $L_1$ diamond boundary at the axes, the optimal solution forces non-informative parameters to **exact zero ($eta_j = 0$)**, achieving automatic feature selection.

## 2. Tri-Model Benchmarking Matrix (OLS vs. Ridge vs. Lasso)

| Model Architecture | Penalty | Active Features | Sparsity (%) | Train R² | Test R² | Adj Test R² | Test RMSE | Test MAE | AIC | BIC |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Unconstrained OLS | None (α = 0) | 43 | 0.0% | 0.8581 | 0.7712 | 0.4902 | 2.143 | 1.544 | 208.46 | 312.72 |
| Ridge Regression | L2 (α* = 15.199) | 43 | 0.0% | 0.856 | 0.7698 | 0.487 | 2.15 | 1.555 | 208.96 | 313.22 |
| Lasso Regression | L1 (α* = 0.1060) | 16 | 62.8% | 0.8453 | 0.7949 | 0.742 | 2.029 | 1.34 | 145.82 | 186.1 |

## 3. Lasso Selected Active Features (16 Retained / 27 Eliminated)

| Feature | Lasso Standardized β | Status | Direction |
| :--- | ---: | ---: | ---: |
| failures | -0.1583 | Retained (Active) ✅ | Negative (-) |
| Fjob_services | -0.1108 | Retained (Active) ✅ | Negative (-) |
| reason_home | -0.0932 | Retained (Active) ✅ | Negative (-) |
| activities | -0.0858 | Retained (Active) ✅ | Negative (-) |
| age | -0.0796 | Retained (Active) ✅ | Negative (-) |
| studytime | -0.0023 | Retained (Active) ✅ | Negative (-) |
| romantic | -0.0012 | Retained (Active) ✅ | Negative (-) |
| health | 0.0001 | Retained (Active) ✅ | Positive (+) |
| higher | 0.0034 | Retained (Active) ✅ | Positive (+) |
| traveltime | 0.0058 | Retained (Active) ✅ | Positive (+) |
| freetime | 0.0455 | Retained (Active) ✅ | Positive (+) |
| Walc | 0.1157 | Retained (Active) ✅ | Positive (+) |
| famrel | 0.178 | Retained (Active) ✅ | Positive (+) |
| absences | 0.2059 | Retained (Active) ✅ | Positive (+) |
| prior_avg_grade | 1.1315 | Retained (Active) ✅ | Positive (+) |
| G2 | 2.9329 | Retained (Active) ✅ | Positive (+) |

## 4. Key Statistical Insights & Parsimony Analysis

1. **Dramatic Sparsity Achievement (62.8% Pruned):**
   - Lasso automatically compressed the feature space from **43 parameters down to 16 essential predictors**, pruning 27 redundant or noisy features without sacrificing predictive performance.

2. **Surge in Adjusted Test $R^2$ (0.7420 vs 0.4902 in OLS):**
   - Because Lasso eliminated 27 degrees of freedom, the Adjusted Test $R^2$ rebounds from **0.4902 to 0.7420**, demonstrating exceptional model parsimony.

3. **Lowest Information Loss (AIC & BIC Winner):**
   - Lasso achieves the lowest AIC (**145.82**) and BIC (**186.10**) of any model tested, proving that sparse feature selection maximizes generalization.

4. **The Core Survival Set:**
   - The predictors that survive Lasso shrinkage are: `G2` (Period 2 Exam), `prior_avg_grade` (Academic momentum), `absences` (Attendance indicator), `famrel` (Family support), and `activities`.
