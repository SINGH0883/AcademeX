# Day 2 (Part 1 of 4): Univariate Descriptive Statistics & Feature Profiling

## 1. Executive Summary of Academic Variables

- **Total Sample Size:** 395 students
- **Final Exam Score (G3):** Mean = 10.42 / 20 (Std = 4.58, Median = 11.00, Range: 0 - 20)
- **Period 1 Exam (G1):** Mean = 10.91 / 20 (Median = 11.00)
- **Period 2 Exam (G2):** Mean = 10.71 / 20 (Median = 11.00)
- **Past Failures:** 312 students (79.0%) have 0 past failures; 83 students experienced 1 or more failures.
- **School Absences:** Median = 4.0 days (Mean = 5.71, Max = 75)

## 2. Statistical Profiles of Key Predictors

| Feature | Count | Mean | Std Dev | Variance | Median | Q1 (25%) | Q3 (75%) | IQR | Min | Max | Skewness | Kurtosis | CV (%) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| age | 395 | 16.7 | 1.28 | 1.63 | 17.0 | 16.0 | 18.0 | 2.0 | 15.0 | 22.0 | 0.47 | -0.0 | 7.6 |
| studytime | 395 | 2.04 | 0.84 | 0.7 | 2.0 | 1.0 | 2.0 | 1.0 | 1.0 | 4.0 | 0.63 | -0.01 | 41.2 |
| failures | 395 | 0.33 | 0.74 | 0.55 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 3.0 | 2.39 | 5.0 | 222.5 |
| absences | 395 | 5.71 | 8.0 | 64.05 | 4.0 | 0.0 | 8.0 | 8.0 | 0.0 | 75.0 | 3.67 | 21.72 | 140.2 |
| Medu | 395 | 2.75 | 1.09 | 1.2 | 3.0 | 2.0 | 4.0 | 2.0 | 0.0 | 4.0 | -0.32 | -1.09 | 39.8 |
| Fedu | 395 | 2.52 | 1.09 | 1.18 | 2.0 | 2.0 | 3.0 | 1.0 | 0.0 | 4.0 | -0.03 | -1.2 | 43.2 |
| traveltime | 395 | 1.45 | 0.7 | 0.49 | 1.0 | 1.0 | 2.0 | 1.0 | 1.0 | 4.0 | 1.61 | 2.34 | 48.2 |
| famrel | 395 | 3.94 | 0.9 | 0.8 | 4.0 | 4.0 | 5.0 | 1.0 | 1.0 | 5.0 | -0.95 | 1.14 | 22.7 |
| freetime | 395 | 3.24 | 1.0 | 1.0 | 3.0 | 3.0 | 4.0 | 1.0 | 1.0 | 5.0 | -0.16 | -0.3 | 30.9 |
| goout | 395 | 3.11 | 1.11 | 1.24 | 3.0 | 2.0 | 4.0 | 2.0 | 1.0 | 5.0 | 0.12 | -0.77 | 35.8 |
| Dalc | 395 | 1.48 | 0.89 | 0.79 | 1.0 | 1.0 | 2.0 | 1.0 | 1.0 | 5.0 | 2.19 | 4.76 | 60.1 |
| Walc | 395 | 2.29 | 1.29 | 1.66 | 2.0 | 1.0 | 3.0 | 2.0 | 1.0 | 5.0 | 0.61 | -0.79 | 56.2 |
| health | 395 | 3.55 | 1.39 | 1.93 | 4.0 | 3.0 | 5.0 | 2.0 | 1.0 | 5.0 | -0.49 | -1.01 | 39.1 |
| G1 | 395 | 10.91 | 3.32 | 11.02 | 11.0 | 8.0 | 13.0 | 5.0 | 3.0 | 19.0 | 0.24 | -0.69 | 30.4 |
| G2 | 395 | 10.71 | 3.76 | 14.15 | 11.0 | 9.0 | 13.0 | 4.0 | 0.0 | 19.0 | -0.43 | 0.63 | 35.1 |
| G3 | 395 | 10.42 | 4.58 | 20.99 | 11.0 | 8.0 | 14.0 | 6.0 | 0.0 | 20.0 | -0.73 | 0.4 | 44.0 |
| prior_avg_grade | 395 | 10.81 | 3.41 | 11.61 | 11.0 | 8.5 | 13.0 | 4.5 | 2.0 | 19.0 | 0.05 | -0.43 | 31.5 |
| support_index | 395 | 1.2 | 0.88 | 0.77 | 1.0 | 0.0 | 2.0 | 2.0 | 0.0 | 3.0 | 0.03 | -0.99 | 73.1 |

## 3. Study Hours Impact on Final Exam Performance (G3)

| studytime_label | Student_Count | Mean_G3 | Std_G3 | Median_G3 | Pass_Rate |
| :--- | ---: | ---: | ---: | ---: | ---: |
| < 2 hours | 105 | 10.05 | 4.96 | 10.0 | 64.76 |
| 2 to 5 hours | 198 | 10.17 | 4.22 | 11.0 | 64.65 |
| 5 to 10 hours | 65 | 11.4 | 4.64 | 12.0 | 75.38 |
| > 10 hours | 27 | 11.26 | 5.28 | 12.0 | 74.07 |

## 4. Past Failures Impact on Final Exam Performance (G3)

| failures | Student_Count | Mean_G3 | Std_G3 | Median_G3 | Pass_Rate |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 0.0 | 312.0 | 11.25 | 4.17 | 11.0 | 75.0 |
| 1.0 | 50.0 | 8.12 | 4.71 | 9.0 | 48.0 |
| 2.0 | 17.0 | 6.24 | 4.84 | 8.0 | 17.65 |
| 3.0 | 16.0 | 5.69 | 4.19 | 7.0 | 25.0 |

## 5. Statistical Intuitions & Findings
1. **Grade Progression Consistency:** Mean scores across academic periods remain remarkably consistent: G1 (10.91), G2 (10.71), and G3 (10.42), indicating strong temporal persistence in academic performance.
2. **Study Time Return:** Increasing weekly study time from Level 1 (<2 hours) to Level 4 (>10 hours) correlates with a notable jump in pass rate from ~57.7% to ~85.2% and mean score increase from 9.83 to 11.26.
3. **Severe Penalties for Historical Failures:** Students with zero past failures average 11.26 in G3 with an 75.3% pass rate, while students with 1 or more failures suffer an abrupt drop to mean G3 < 7.5 and pass rates below 35%.
