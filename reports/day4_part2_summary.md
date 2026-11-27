# Day 4 (Part 2 of 4): Ridge Regression (L2 Regularization) & Alpha CV

## 1. Mathematical Formulation of L2 Regularization

Ridge Regression modifies the Ordinary Least Squares objective by adding a quadratic penalty proportional to the Euclidean $L_2$ norm of the parameter vector:

$$\min_\beta \left\{ \frac{1}{2n} \|y - X \beta\|_2^2 + \frac{\alpha}{2} \|\beta\|_2^2 \right\} = \min_\beta \left\{ \frac{1}{2n} \sum_{i=1}^n (y_i - x_i^T \beta)^2 + \frac{\alpha}{2} \sum_{j=1}^p \beta_j^2 \right\}

The analytical closed-form solution is:

$$\hat{\beta}_{\text{ridge}} = (X^T X + n\alpha I)^{-1} X^T y$$

Because $n\alpha I$ adds positive values to the diagonal of $X^T X$, the matrix is guaranteed to be strictly non-singular and well-conditioned, directly resolving the multicollinearity detected between $G1-G2$ and demographic indicators in Day 3.

## 2. Comparative Performance: OLS vs. Ridge Regression

| Model Architecture | Regularization (Penalty) | Train R² | Test R² | Adj Test R² | Test RMSE | Test MAE | Max |β| | Mean |β| |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Unconstrained OLS (Full 43 Features) | None (α = 0) | 0.8581 | 0.7712 | 0.4902 | 2.143 | 1.544 | 2.718 | 0.203 |
| Ridge Regression (L2 Shrinkage) | L2 (α* = 15.199) | 0.856 | 0.7698 | 0.487 | 2.15 | 1.555 | 2.325 | 0.193 |

## 3. Top 10 Features Under Ridge Regularization

| Feature | Ridge Standardized β | OLS Standardized β | Shrinkage Ratio (Ridge/OLS) |
| :--- | ---: | ---: | ---: |
| G2 | 2.3246 | 2.7184 | 0.8551 |
| prior_avg_grade | 1.4439 | 1.4908 | 0.9685 |
| absences | 0.3453 | 0.3709 | 0.931 |
| G1 | 0.3086 | -0.0483 | -6.3885 |
| Walc | 0.2865 | 0.3025 | 0.9471 |
| famrel | 0.2495 | 0.2765 | 0.9026 |
| activities | -0.2399 | -0.2496 | 0.9612 |
| failures | -0.2245 | -0.22 | 1.0209 |
| age | -0.205 | -0.2091 | 0.9806 |
| Medu | 0.1963 | 0.2036 | 0.9642 |

## 4. Key Statistical Takeaways & Bias-Variance Tradeoff

1. **Optimal Penalty Parameter ($\alpha^* = 15.199$):** 5-fold cross-validation identified an optimal penalty of $\alpha = 15.199$, striking the optimal balance between bias introduction and variance reduction.
2. **Mitigation of Parameter Variance:** The maximum coefficient magnitude was compressed from 2.718 in OLS to 2.325 in Ridge, and mean coefficient magnitude was reduced, dampening coefficient sensitivity.
3. **Continuous vs. Sparse Shrinkage:** Ridge continuously pulls all coefficients towards zero without setting any exactly to zero. Features like `Fjob_health` that had unreliably high slopes in OLS were stabilized. Next, in Part 3, Lasso will test whether sparse elimination of these zero-signal regressors yields superior generalization.
