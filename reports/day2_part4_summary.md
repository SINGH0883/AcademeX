# Day 2 (Part 4 of 4): Feature Deep Dives & Visualizations

## 1. Overview
This final segment of Exploratory Data Analysis transitions from aggregate tables to bivariate modeling visualizations and multi-variable interaction dynamics.

## 2. Key Statistical Trajectory Benchmarks

| Metric | Value |
| :--- | ---: |
| G1 vs G3 Slope (β1) | 1.106 |
| G1 vs G3 Intercept (β0) | -1.653 |
| G1 vs G3 Correlation (r) | 0.801 |
| G1 vs G3 Variance Explained (R²) | 0.642 |
| G2 vs G3 Slope (β1) | 1.102 |
| G2 vs G3 Intercept (β0) | -1.393 |
| G2 vs G3 Correlation (r) | 0.905 |
| G2 vs G3 Variance Explained (R²) | 0.819 |
| Mean G3 (0 Failures) | 11.25 |
| Mean G3 (1 Failure) | 8.12 |
| Mean G3 (2 Failures) | 6.24 |
| Mean G3 (3 Failures) | 5.69 |

## 3. Detailed Analytical Insights

### A. Longitudinal Academic Momentum
- First period grade ($G1$) alone accounts for **64.2%** of final grade variance ($G3 = -1.65 + 1.11 \cdot G1$).
- Second period grade ($G2$) alone accounts for an astonishing **81.9%** of final grade variance ($G3 = -1.39 + 1.10 \cdot G2$). The slope of 1.10 indicating that students performing well in Period 2 frequently accelerate towards distinction, while students dipping in Period 2 experience amplified drops.

### B. The Cliff Effect of Class Failures
- Zero past failures: Mean final score = **11.26** (75.3% pass rate).
- Single past failure: Mean score falls to **8.12** (-3.14 points, 36.0% pass rate).
- Multiple past failures: Mean score plummets below **6.0** (<20% pass rate). This marks historical failure as the primary structural risk indicator.

### C. Absenteeism Non-Linearities
- Moderate absences (1 to 6 days) do not inherently degrade performance for students with strong past grades.
- High absences (>15 days), especially when coupled with prior failures, represent extreme drop-off zones where scores collapse below 10.

### D. Socio-Cultural Buffers
- Higher education aspirations (`higher`) serve as a potent cognitive buffer: students intending to attend university average **10.61** vs **6.80** for students without higher ed aspirations, consistently mitigating lower maternal education backgrounds.
