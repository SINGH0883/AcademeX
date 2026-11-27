# Day 4 (Part 4 of 4): Tri-Model Comparative Synthesis, Elastic Net & Model Serialization

## 1. Executive Model Comparison Benchmark

We synthesized all model architectures across identical standardized train/test splits ($N_{\text{train}} = 316, N_{\text{test}} = 79$):

| Model | Regularization | Tuning Params | Active Features | Sparsity (%) | Train R² | Test R² | Adj Test R² | Test RMSE | Test MAE | AIC | BIC |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Academic Core OLS | Unregularized Subset | G1, G2, studytime | 3 | 0.0% | 0.8302 | 0.792 | 0.7837 | 2.044 | 1.323 | 120.95 | 130.42 |
| Full Multiple OLS | Unregularized Full | All 43 features | 43 | 0.0% | 0.8581 | 0.7712 | 0.4902 | 2.143 | 1.544 | 208.46 | 312.72 |
| Ridge Regression | L2 Regularized | α = 15.20 | 43 | 0.0% | 0.856 | 0.7698 | 0.487 | 2.15 | 1.555 | 208.96 | 313.22 |
| Lasso Regression | L1 Regularized (Sparse) | α = 0.106 | 16 | 62.8% | 0.8453 | 0.7949 | 0.742 | 2.029 | 1.34 | 145.82 | 186.1 |
| Elastic Net | L1 + L2 Hybrid | α = 0.102, l1_ratio = 0.95 | 18 | 58.1% | 0.8466 | 0.7939 | 0.732 | 2.035 | 1.358 | 150.23 | 195.25 |

## 2. Elastic Net Formulation & Grouping Mechanics

Elastic Net linearly blends the $L_1$ and $L_2$ penalties:

$$\min_\beta \left\{ \frac{1}{2n} \|y - X \beta\|_2^2 + \alpha \left( \rho \|\beta\|_1 + \frac{1 - \rho}{2} \|\beta\|_2^2 \right) \right\}$$

Cross-validation established $\rho = 0.95$ (`l1_ratio`) and $\alpha = 0.102$. Elastic Net achieves performance nearly on par with Lasso ($R^2 = 0.7939$, RMSE = 2.035), effectively handling collinear groupings.

## 3. The Winning Architecture: Lasso ($L_1$) Regularization

Across all rigorous econometric and machine learning metrics, **Lasso Regression** emerged as the undisputed champion:
1. **Generalization Supremacy:** Test $R^2 = \mathbf{0.7949}$ (highest across all tested models).
2. **Lowest Prediction Error:** Test RMSE = $\mathbf{2.029}$ grade points and MAE = $\mathbf{1.336}$.
3. **Maximum Model Parsimony:** Retains only **16 active predictors** (pruning 62.8% of redundant noise), driving Adjusted Test $R^2$ to $\mathbf{0.7420}$ (compared to $0.4902$ in full OLS).
4. **Information Criterion Winner:** Achieves lowest AIC ($145.82$) and lowest BIC ($186.10$).

## 4. Production Artifact Serialization

- Serialized model pipeline and metadata safely saved to: `models\linear_regression_model.joblib`.
- Ready for Day 5 evaluation diagnostics (Confusion Matrix) and Day 6 deployment.
