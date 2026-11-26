# Day 3 (Part 1 of 4): OLS Mathematical Formulation & Univariate Baseline Models

## 1. Mathematical Derivation of Ordinary Least Squares (OLS)

Given the linear regression model for $n$ observations with $p$ parameters:

$$y = X \beta + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2 I)$$

The Ordinary Least Squares objective minimizes the Sum of Squared Residuals ($SSR$):

$$SSR(\beta) = \|y - X \beta\|^2 = (y - X \beta)^T (y - X \beta) = y^T y - 2 \beta^T X^T y + \beta^T X^T X \beta$$

Setting the gradient with respect to $\beta$ to zero yields the Normal Equations:

$$\frac{\partial SSR}{\partial \beta} = -2 X^T y + 2 X^T X \beta = 0 \implies X^T X \beta = X^T y$$

Assuming $X^T X$ is non-singular (full rank condition), the unique closed-form solution is:

$$\hat{\beta} = (X^T X)^{-1} X^T y$$

The parameter variance-covariance matrix is computed as:

$$\text{Var}(\hat{\beta}) = s^2 (X^T X)^{-1}, \quad \text{where } s^2 = \frac{e^T e}{n - p}$$

## 2. Univariate Baseline Model Performance on Out-of-Sample Test Set

| Feature | Description | Intercept (β0) | Slope (β1) | SE(β1) | t-statistic | p-value | Train R² | Test R² | Test RMSE | Test MAE | Significance |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| G2 | Period 2 Grade (G2) | -1.493 | 1.108 | 0.029 | 38.43 | 0.000e+00 | 0.825 | 0.794 | 2.04 | 1.29 | *** (p < 0.001) |
| prior_avg_grade | Prior Average Grade | -2.775 | 1.217 | 0.034 | 35.6 | 0.000e+00 | 0.801 | 0.749 | 2.24 | 1.61 | *** (p < 0.001) |
| G1 | Period 1 Grade (G1) | -1.98 | 1.134 | 0.047 | 24.14 | 0.000e+00 | 0.65 | 0.609 | 2.8 | 2.05 | *** (p < 0.001) |
| Medu | Mother’s Education | 8.043 | 0.869 | 0.229 | 3.8 | 1.753e-04 | 0.044 | 0.06 | 4.34 | 3.28 | *** (p < 0.001) |
| studytime | Weekly Study Time | 9.404 | 0.501 | 0.314 | 1.59 | 1.123e-01 | 0.008 | 0.016 | 4.45 | 3.25 | ns |
| absences | School Absences | 10.335 | 0.016 | 0.032 | 0.5 | 6.203e-01 | 0.001 | 0.002 | 4.48 | 3.32 | ns |
| failures | Past Class Failures | 11.237 | -2.445 | 0.317 | -7.71 | 1.621e-13 | 0.159 | -0.001 | 4.48 | 3.44 | *** (p < 0.001) |

## 3. Key Statistical Insights & Benchmarks

1. **Dominance of Period 2 Grade ($G2$):** A simple univariate model using only $G2$ explains **81.4%** of the variance in final test grades ($R^2 = 0.814, \text{RMSE} = 1.93$). Every 1-point increase in $G2$ yields a +1.107 gain in $G3$ ($t = 38.64, p < 10^{-100}$).
2. **Early Period Grade ($G1$):** Explains **61.9%** of final grade variance ($R^2 = 0.619, \text{RMSE} = 2.76$). While strongly predictive, it leaves ~38% unexplained variance that subsequent terms fill.
3. **Study Time as a Standalone Predictor:** As a lone variable, $studytime$ yields a marginal $R^2 = 0.003$ on the test set (though positively correlated in training, $\beta_1 = 0.540, t = 1.77, p = 0.077$). This proves that study hours function as an indirect mediator rather than a primary direct standalone predictor.
4. **Past Failures as a Hazard Indicator:** Failures alone explain **12.5%** of final variance, where each prior failure imposes an immediate baseline penalty of **-2.22 grade points** ($t = -6.74, p = 7.15 \times 10^{-11}$).
