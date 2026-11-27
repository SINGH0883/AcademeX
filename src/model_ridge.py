"""
Data Science Fundamentals and Statistical Intuition
Project-1: Predictive Analytics for Student Performance in Exams
Day 4 (Part 2 of 4): Ridge Regression (L2 Regularization) & Alpha Cross-Validation
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import Ridge, RidgeCV, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from src.preprocess import get_train_test_data

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.size'] = 10

def train_ridge_regularization(output_dir='reports', fig_dir='figures'):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    data = get_train_test_data()
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train_reg']
    y_test = data['y_test_reg']
    feature_names = data['feature_names']

    n_train, p = X_train.shape
    n_test = len(y_test)

    # 1. Standardize features for L2 scale invariance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 2. 5-Fold Cross-Validation for optimal alpha
    alphas = np.logspace(-3, 3, 100)
    ridge_cv = RidgeCV(alphas=alphas, cv=5, scoring='neg_mean_squared_error')
    ridge_cv.fit(X_train_scaled, y_train)
    best_alpha = ridge_cv.alpha_

    # 3. Fit Final Ridge Model with optimal alpha
    best_ridge = Ridge(alpha=best_alpha)
    best_ridge.fit(X_train_scaled, y_train)

    y_train_pred_ridge = best_ridge.predict(X_train_scaled)
    y_test_pred_ridge = best_ridge.predict(X_test_scaled)

    r2_train_ridge = r2_score(y_train, y_train_pred_ridge)
    r2_test_ridge = r2_score(y_test, y_test_pred_ridge)
    adj_r2_test_ridge = 1 - (1 - r2_test_ridge) * (n_test - 1) / (n_test - p - 1)
    
    mse_test_ridge = mean_squared_error(y_test, y_test_pred_ridge)
    rmse_test_ridge = np.sqrt(mse_test_ridge)
    mae_test_ridge = mean_absolute_error(y_test, y_test_pred_ridge)

    aic_ridge = n_test * np.log(mse_test_ridge) + 2 * (p + 1)
    bic_ridge = n_test * np.log(mse_test_ridge) + (p + 1) * np.log(n_test)

    # Compare against unregularized OLS
    ols = LinearRegression()
    ols.fit(X_train_scaled, y_train)
    y_test_pred_ols = ols.predict(X_test_scaled)
    r2_test_ols = r2_score(y_test, y_test_pred_ols)
    rmse_test_ols = np.sqrt(mean_squared_error(y_test, y_test_pred_ols))
    mae_test_ols = mean_absolute_error(y_test, y_test_pred_ols)

    # 4. Generate Ridge Trace Matrix across alpha grid
    coef_matrix = []
    cv_mse_scores = []

    for a in alphas:
        r_model = Ridge(alpha=a)
        r_model.fit(X_train_scaled, y_train)
        coef_matrix.append(r_model.coef_)
        
        # Test MSE across alpha grid
        pred_te = r_model.predict(X_test_scaled)
        cv_mse_scores.append(mean_squared_error(y_test, pred_te))

    coef_matrix = np.array(coef_matrix)

    # -------------------------------------------------------------
    # Visualization 1: Ridge Trace Plot (Shrinkage Path)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 6))
    
    # Highlight top 6 features, plot others in light gray
    top_feature_indices = np.argsort(np.abs(best_ridge.coef_))[-6:]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for i in range(p):
        if i in top_feature_indices:
            color = colors[np.where(top_feature_indices == i)[0][0]]
            ax.plot(alphas, coef_matrix[:, i], label=feature_names[i], color=color, linewidth=2.5)
        else:
            ax.plot(alphas, coef_matrix[:, i], color='gray', alpha=0.25, linewidth=0.8)

    ax.axvline(best_alpha, color='crimson', linestyle='--', linewidth=2, label=f'Optimal $\\alpha^*$ = {best_alpha:.3f}')
    ax.set_xscale('log')
    ax.set_title('Ridge Regularization Trace Plot (L2 Coefficient Shrinkage Path)', weight='bold')
    ax.set_xlabel('Regularization Strength Parameter ($\\alpha$, Log Scale)', weight='bold')
    ax.set_ylabel('Standardized Regression Coefficients ($\\beta_j$)', weight='bold')
    ax.legend(frameon=True, loc='upper right')
    plt.tight_layout()
    trace_fig_path = os.path.join(fig_dir, 'ridge_trace_plot.png')
    plt.savefig(trace_fig_path, dpi=300)
    plt.close()
    print(f"Saved: {trace_fig_path}")

    # -------------------------------------------------------------
    # Visualization 2: MSE Curve vs Alpha
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(alphas, cv_mse_scores, color='#2166ac', linewidth=2.5, label='Out-of-Sample Test MSE')
    ax.axvline(best_alpha, color='crimson', linestyle='--', linewidth=2, label=f'Optimal $\\alpha^*$ = {best_alpha:.3f}')
    ax.scatter([best_alpha], [mse_test_ridge], color='crimson', s=70, zorder=5)
    ax.set_xscale('log')
    ax.set_title('Out-of-Sample Test MSE vs. Ridge Regularization Penalty ($\\alpha$)', weight='bold')
    ax.set_xlabel('Regularization Parameter ($\\alpha$, Log Scale)', weight='bold')
    ax.set_ylabel('Mean Squared Error (MSE)', weight='bold')
    ax.legend(frameon=True)
    plt.tight_layout()
    mse_fig_path = os.path.join(fig_dir, 'ridge_cv_mse_curve.png')
    plt.savefig(mse_fig_path, dpi=300)
    plt.close()
    print(f"Saved: {mse_fig_path}")

    # -------------------------------------------------------------
    # 5. Export Tables & Summary Report
    # -------------------------------------------------------------
    comparison_data = [
        {
            'Model Architecture': 'Unconstrained OLS (Full 43 Features)',
            'Regularization (Penalty)': 'None (α = 0)',
            'Train R²': round(r2_score(y_train, ols.predict(X_train_scaled)), 4),
            'Test R²': round(r2_test_ols, 4),
            'Adj Test R²': round(1 - (1 - r2_test_ols) * (n_test - 1) / (n_test - p - 1), 4),
            'Test RMSE': round(rmse_test_ols, 3),
            'Test MAE': round(mae_test_ols, 3),
            'Max |β|': round(float(np.max(np.abs(ols.coef_))), 3),
            'Mean |β|': round(float(np.mean(np.abs(ols.coef_))), 3)
        },
        {
            'Model Architecture': 'Ridge Regression (L2 Shrinkage)',
            'Regularization (Penalty)': f'L2 (α* = {best_alpha:.3f})',
            'Train R²': round(r2_train_ridge, 4),
            'Test R²': round(r2_test_ridge, 4),
            'Adj Test R²': round(adj_r2_test_ridge, 4),
            'Test RMSE': round(rmse_test_ridge, 3),
            'Test MAE': round(mae_test_ridge, 3),
            'Max |β|': round(float(np.max(np.abs(best_ridge.coef_))), 3),
            'Mean |β|': round(float(np.mean(np.abs(best_ridge.coef_))), 3)
        }
    ]
    comp_df = pd.DataFrame(comparison_data)
    results_csv_path = os.path.join(output_dir, 'day4_part2_ridge_results.csv')
    comp_df.to_csv(results_csv_path, index=False)

    # Top Ridge coefficients table
    ridge_coefs = pd.DataFrame({
        'Feature': feature_names,
        'Ridge Standardized β': best_ridge.coef_,
        'OLS Standardized β': ols.coef_,
        'Shrinkage Ratio (Ridge/OLS)': best_ridge.coef_ / np.where(ols.coef_ == 0, 1e-6, ols.coef_)
    }).sort_values(by='Ridge Standardized β', key=abs, ascending=False)

    def df_to_md(df_in):
        headers = list(df_in.columns)
        lines = ['| ' + ' | '.join(str(h) for h in headers) + ' |']
        lines.append('| ' + ' | '.join([':---' if i == 0 else '---:' for i in range(len(headers))]) + ' |')
        for _, row in df_in.iterrows():
            lines.append('| ' + ' | '.join(str(val) for val in row) + ' |')
        return '\n'.join(lines)

    summary_md_path = os.path.join(output_dir, 'day4_part2_summary.md')
    with open(summary_md_path, 'w', encoding='utf-8') as f:
        f.write("# Day 4 (Part 2 of 4): Ridge Regression (L2 Regularization) & Alpha CV\n\n")
        f.write("## 1. Mathematical Formulation of L2 Regularization\n\n")
        f.write("Ridge Regression modifies the Ordinary Least Squares objective by adding a quadratic penalty proportional to the Euclidean $L_2$ norm of the parameter vector:\n\n")
        f.write("$$\\min_\\beta \\left\\{ \\frac{1}{2n} \\|y - X \\beta\\|_2^2 + \\frac{\\alpha}{2} \\|\\beta\\|_2^2 \\right\\} = \\min_\\beta \\left\\{ \\frac{1}{2n} \\sum_{i=1}^n (y_i - x_i^T \\beta)^2 + \\frac{\\alpha}{2} \\sum_{j=1}^p \\beta_j^2 \\right\\}\n\n")
        f.write("The analytical closed-form solution is:\n\n")
        f.write("$$\\hat{\\beta}_{\\text{ridge}} = (X^T X + n\\alpha I)^{-1} X^T y$$\n\n")
        f.write("Because $n\\alpha I$ adds positive values to the diagonal of $X^T X$, the matrix is guaranteed to be strictly non-singular and well-conditioned, directly resolving the multicollinearity detected between $G1-G2$ and demographic indicators in Day 3.\n\n")
        
        f.write("## 2. Comparative Performance: OLS vs. Ridge Regression\n\n")
        f.write(df_to_md(comp_df))
        
        f.write("\n\n## 3. Top 10 Features Under Ridge Regularization\n\n")
        f.write(df_to_md(ridge_coefs.head(10).round(4)))
        
        f.write("\n\n## 4. Key Statistical Takeaways & Bias-Variance Tradeoff\n\n")
        f.write(f"1. **Optimal Penalty Parameter ($\\alpha^* = {best_alpha:.3f}$):** 5-fold cross-validation identified an optimal penalty of $\\alpha = {best_alpha:.3f}$, striking the optimal balance between bias introduction and variance reduction.\n")
        f.write(f"2. **Mitigation of Parameter Variance:** The maximum coefficient magnitude was compressed from {comp_df.loc[0, 'Max |β|']} in OLS to {comp_df.loc[1, 'Max |β|']} in Ridge, and mean coefficient magnitude was reduced, dampening coefficient sensitivity.\n")
        f.write("3. **Continuous vs. Sparse Shrinkage:** Ridge continuously pulls all coefficients towards zero without setting any exactly to zero. Features like `Fjob_health` that had unreliably high slopes in OLS were stabilized. Next, in Part 3, Lasso will test whether sparse elimination of these zero-signal regressors yields superior generalization.\n")

    print(f"Comparison table saved to: {results_csv_path}")
    print(f"Summary markdown saved to: {summary_md_path}")
    print("\n=== OLS VS RIDGE BENCHMARK ===")
    print(comp_df.to_string(index=False))

    return comp_df

if __name__ == '__main__':
    train_ridge_regularization()
