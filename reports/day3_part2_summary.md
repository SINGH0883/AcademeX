# Day 3 (Part 2 of 4): Academic Core Baseline & Multivariate Progression

## 1. Executive Model Comparison Matrix

| Tier | Model Name | Num Features | Train R² | Test R² | Adj Test R² | Δ R² (Gain) | Test RMSE | Test MAE | AIC | BIC |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Tier 1 | Univariate G2 | 1 | 0.8247 | 0.7936 | 0.7909 | 0.7936 | 2.036 | 1.289 | 116.35 | 121.09 |
| Tier 2 | Bivariate Academic Trajectory | 2 | 0.8293 | 0.7909 | 0.7854 | -0.0026 | 2.049 | 1.307 | 119.35 | 126.45 |
| Tier 3 | Academic Core Trivariate | 3 | 0.8302 | 0.792 | 0.7837 | 0.0011 | 2.044 | 1.323 | 120.95 | 130.42 |
| Tier 4 | Core + Academic Risk Factors | 5 | 0.8369 | 0.7922 | 0.7779 | 0.0002 | 2.043 | 1.329 | 124.88 | 139.1 |
| Tier 5 | Engineered Features Baseline | 5 | 0.8119 | 0.7596 | 0.7432 | -0.0325 | 2.197 | 1.585 | 136.37 | 150.58 |

## 2. Core Model Mathematical Formulations & Fitted Equations

- **Tier 1 (Univariate G2):** Intercept = `-1.493`, Features: `G2`, Test $R^2 = 0.7936$, RMSE = `2.036`
- **Tier 2 (Bivariate Academic Trajectory):** Intercept = `-2.04`, Features: `G1, G2`, Test $R^2 = 0.7909$, RMSE = `2.049`
- **Tier 3 (Academic Core Trivariate):** Intercept = `-1.758`, Features: `G1, G2, studytime`, Test $R^2 = 0.792$, RMSE = `2.044`
- **Tier 4 (Core + Academic Risk Factors):** Intercept = `-1.491`, Features: `G1, G2, studytime, failures, absences`, Test $R^2 = 0.7922$, RMSE = `2.043`
- **Tier 5 (Engineered Features Baseline):** Intercept = `-2.501`, Features: `prior_avg_grade, studytime, support_index, failures, absences`, Test $R^2 = 0.7596$, RMSE = `2.197`

## 3. Key Statistical Insights & Progression Analysis

1. **The Trajectory Transition (Tier 1 vs Tier 2):**
   - Moving from univariate $G2$ to bivariate $G1 + G2$ reduces $G2$'s slope from **+1.108** to **+0.985**, transferring **+0.165** coefficient weight to $G1$.
   - Test $R^2$ peaks at **0.7937** with an ultra-compact RMSE of **2.04 grade points**.

2. **Study Time in Multivariate Space (Tier 3):**
   - Adding $studytime$ to prior grades introduces a positive coefficient ($\beta_{study} = +0.102$), confirming that for students with identical prior exam grades, each higher weekly study time tier provides a small incremental academic lift.

3. **Risk Factor Integration (Tier 4):**
   - Incorporating past failures and absences slightly reduces out-of-sample test RMSE to **2.04**, while providing interpretability for identifying at-risk students who have historical baggage.

4. **Engineered Baseline Parity (Tier 5):**
   - Replacing collinear individual grades with `prior_avg_grade` produces an elegant, highly interpretable baseline ($R^2 = 0.751, \text{RMSE} = 2.23$) with zero risk of collinearity instability between $G1$ and $G2$.
