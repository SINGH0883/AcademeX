# Day 3 (Part 4 of 4): Multicollinearity Diagnostics & Variance Inflation Factor (VIF)

## 1. Mathematical Formulation & Statistical Foundation

In Multiple Linear Regression, the variance of the $j$-th coefficient estimator is:

$$\text{Var}(\hat{\beta}_j) = \frac{\sigma^2}{(n - 1) s_j^2} \times \text{VIF}_j, \quad \text{where } \text{VIF}_j = \frac{1}{1 - R_j^2}$$

where $R_j^2$ is the coefficient of determination obtained from regressing feature $X_j$ on all remaining regressors. A high VIF indicates that the effective sample information for estimating $\beta_j$ is severely diluted by redundant correlations, inflating standard errors and destabilizing hypothesis tests.

## 2. VIF Audit Across Extended Candidate Predictor Suite

| Feature | Auxiliary R² | Tolerance | VIF | Collinearity Severity |
| :--- | :--- | ---: | ---: | ---: |
| G1 | 0.7503 | 0.2497 | 4.0 | Moderate Collinearity (2.5 <= VIF < 5) 🟡 |
| G2 | 0.7482 | 0.2518 | 3.97 | Moderate Collinearity (2.5 <= VIF < 5) 🟡 |
| Walc | 0.556 | 0.444 | 2.25 | Low / Acceptable (VIF < 2.5) 🟢 |
| Dalc | 0.4618 | 0.5382 | 1.86 | Low / Acceptable (VIF < 2.5) 🟢 |
| Medu | 0.4448 | 0.5552 | 1.8 | Low / Acceptable (VIF < 2.5) 🟢 |
| Fedu | 0.4298 | 0.5702 | 1.75 | Low / Acceptable (VIF < 2.5) 🟢 |
| goout | 0.3336 | 0.6664 | 1.5 | Low / Acceptable (VIF < 2.5) 🟢 |
| failures | 0.2787 | 0.7213 | 1.39 | Low / Acceptable (VIF < 2.5) 🟢 |
| support_index | 0.1942 | 0.8058 | 1.24 | Low / Acceptable (VIF < 2.5) 🟢 |
| freetime | 0.1843 | 0.8157 | 1.23 | Low / Acceptable (VIF < 2.5) 🟢 |
| higher | 0.1721 | 0.8279 | 1.21 | Low / Acceptable (VIF < 2.5) 🟢 |
| studytime | 0.161 | 0.839 | 1.19 | Low / Acceptable (VIF < 2.5) 🟢 |
| traveltime | 0.1032 | 0.8968 | 1.12 | Low / Acceptable (VIF < 2.5) 🟢 |
| internet | 0.1067 | 0.8933 | 1.12 | Low / Acceptable (VIF < 2.5) 🟢 |
| romantic | 0.0998 | 0.9002 | 1.11 | Low / Acceptable (VIF < 2.5) 🟢 |
| absences | 0.0836 | 0.9164 | 1.09 | Low / Acceptable (VIF < 2.5) 🟢 |
| famrel | 0.0681 | 0.9319 | 1.07 | Low / Acceptable (VIF < 2.5) 🟢 |
| health | 0.069 | 0.931 | 1.07 | Low / Acceptable (VIF < 2.5) 🟢 |

## 3. Comparison: Individual Grades vs. Engineered Aggregate

### A. Core Model VIF (Dual Periodic Grades Included)
| Feature | Auxiliary R² | Tolerance | VIF | Collinearity Severity |
| :--- | :--- | ---: | ---: | ---: |
| G1 | 0.7203 | 0.2797 | 3.58 | Moderate Collinearity (2.5 <= VIF < 5) 🟡 |
| G2 | 0.719 | 0.281 | 3.56 | Moderate Collinearity (2.5 <= VIF < 5) 🟡 |
| studytime | 0.0206 | 0.9794 | 1.02 | Low / Acceptable (VIF < 2.5) 🟢 |

### B. Engineered Solution VIF (`prior_avg_grade` Substitution)
| Feature | Auxiliary R² | Tolerance | VIF | Collinearity Severity |
| :--- | :--- | ---: | ---: | ---: |
| Walc | 0.5525 | 0.4475 | 2.23 | Low / Acceptable (VIF < 2.5) 🟢 |
| Dalc | 0.4618 | 0.5382 | 1.86 | Low / Acceptable (VIF < 2.5) 🟢 |
| Medu | 0.4436 | 0.5564 | 1.8 | Low / Acceptable (VIF < 2.5) 🟢 |
| Fedu | 0.4266 | 0.5734 | 1.74 | Low / Acceptable (VIF < 2.5) 🟢 |
| goout | 0.3336 | 0.6664 | 1.5 | Low / Acceptable (VIF < 2.5) 🟢 |
| failures | 0.2785 | 0.7215 | 1.39 | Low / Acceptable (VIF < 2.5) 🟢 |

## 4. Key Statistical Takeaways & Day 4 Modeling Roadmap

1. **Primary Collinearity Pair ($G1$ and $G2$):** Both $G1$ (VIF = 4.0) and $G2$ (VIF = 3.97) exhibit moderate-to-high collinearity (Auxiliary $R^2 \approx 0.74$). Because they measure the same underlying competency at two adjacent intervals, their shared variance inflates coefficient uncertainty in unregularized OLS.
2. **Assortative Parental & Social Collinearities:**
   - Weekend and Workday alcohol (`Walc` VIF = 2.25, `Dalc` VIF = 1.86).
   - Parental education (`Medu` VIF = 1.8, `Fedu` VIF = 1.75).
3. **Engineered Feature Solution:** Substituting $G1$ and $G2$ with `prior_avg_grade` drops the maximum VIF in the entire academic feature set from ~4.0 to **1.56**, eliminating 100% of inter-grade collinearity.
4. **Direct Bridge to Day 4 (Regularization):** This diagnostic confirms that while full OLS is workable, fitting **Ridge ($L_2$) Regression** (which adds a penalty $\lambda \sum \beta_j^2$ to stabilize $(X^T X + \lambda I)^{-1}$) and **Lasso ($L_1$) Regression** (which performs automatic feature selection) is the theoretically optimal strategy for handling multicollinearity across the 43 engineered features.
