"""
Data Science Fundamentals and Statistical Intuition
Project-1: Predictive Analytics for Student Performance in Exams
Module: Model Building, Evaluation, and Diagnostic Visualizations
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score
)
from src.preprocess import get_train_test_data

def train_and_evaluate_models(output_dir='figures', model_dir='models'):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    data = get_train_test_data()
    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train_reg'], data['y_test_reg']
    y_train_cls, y_test_cls = data['y_train_cls'], data['y_test_cls']
    feature_names = data['feature_names']

    print("--- 1. Training Multiple Linear Regression ---")
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # Predictions
    y_train_pred = lr.predict(X_train)
    y_test_pred = lr.predict(X_test)

    # Post-process predictions to respect valid grade bounds [0, 20]
    y_test_pred_clipped = np.clip(y_test_pred, 0, 20)

    # Regression Metrics
    r2 = r2_score(y_test, y_test_pred)
    n = len(y_test)
    p = X_test.shape[1]
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    mse = mean_squared_error(y_test, y_test_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_test_pred)

    print(f"Regression Performance on Test Set:")
    print(f"  R-squared (R²):        {r2:.4f}")
    print(f"  Adjusted R-squared:    {adj_r2:.4f}")
    print(f"  Mean Squared Error:    {mse:.4f}")
    print(f"  Root Mean Squared Err: {rmse:.4f}")
    print(f"  Mean Absolute Error:   {mae:.4f}")

    # Baseline comparison: Univariate Linear Regression on Prior Grades & Study Time
    lr_base = LinearRegression()
    lr_base.fit(X_train[['G1', 'G2', 'studytime']], y_train)
    base_pred = lr_base.predict(X_test[['G1', 'G2', 'studytime']])
    base_r2 = r2_score(y_test, base_pred)
    base_rmse = np.sqrt(mean_squared_error(y_test, base_pred))
    print(f"\nBaseline (G1, G2, studytime) R²: {base_r2:.4f}, RMSE: {base_rmse:.4f}")

    # Ridge & Lasso Regularization comparison
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    ridge_pred = ridge.predict(X_test)
    ridge_r2 = r2_score(y_test, ridge_pred)

    lasso = Lasso(alpha=0.05)
    lasso.fit(X_train, y_train)
    lasso_pred = lasso.predict(X_test)
    lasso_r2 = r2_score(y_test, lasso_pred)
    print(f"Ridge Regression R²: {ridge_r2:.4f} | Lasso Regression R²: {lasso_r2:.4f}")

    # --- 2. Classification Evaluation via Passing Threshold (Grade >= 10) ---
    print("\n--- 2. Classification Evaluation (Pass/Fail Threshold >= 10) ---")
    y_test_pred_cls = (y_test_pred_clipped >= 10).astype(int)
    cm = confusion_matrix(y_test_cls, y_test_pred_cls)
    acc = accuracy_score(y_test_cls, y_test_pred_cls)
    prec = precision_score(y_test_cls, y_test_pred_cls)
    rec = recall_score(y_test_cls, y_test_pred_cls)
    f1 = f1_score(y_test_cls, y_test_pred_cls)

    print(f"Classification Performance:")
    print(f"  Accuracy:  {acc:.4f} ({acc*100:.1f}%)")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)

    # --- 3. Save Model and Metadata ---
    model_bundle = {
        'model': lr,
        'feature_names': feature_names,
        'metrics': {
            'r2': float(r2),
            'adj_r2': float(adj_r2),
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'base_r2': float(base_r2),
            'ridge_r2': float(ridge_r2),
            'lasso_r2': float(lasso_r2),
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist()
        }
    }
    model_path = os.path.join(model_dir, 'linear_regression_model.joblib')
    joblib.dump(model_bundle, model_path)
    print(f"\nTrained model and metrics saved to: {model_path}")

    # --- 4. Diagnostic & Evaluation Charts ---

    # Chart 1: Actual vs Predicted Parity Plot
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, y_test_pred_clipped, color='#1f77b4', edgecolors='black', alpha=0.7, s=60, label='Test Observations')
    min_val = min(y_test.min(), y_test_pred_clipped.min())
    max_val = max(y_test.max(), y_test_pred_clipped.max())
    ax.plot([0, 20], [0, 20], color='crimson', linestyle='--', linewidth=2, label='Perfect Prediction (y = x)')
    ax.set_title(f'Actual vs Predicted Final Exam Grade (G3)\nR² = {r2:.3f}, RMSE = {rmse:.2f}')
    ax.set_xlabel('Actual Exam Grade (G3)')
    ax.set_ylabel('Predicted Exam Grade (G3)')
    ax.set_xlim(-1, 21)
    ax.set_ylim(-1, 21)
    ax.legend(frameon=True)
    plt.tight_layout()
    parity_path = os.path.join(output_dir, 'actual_vs_predicted.png')
    plt.savefig(parity_path, dpi=300)
    plt.close()
    print(f"Saved: {parity_path}")

    # Chart 2: Residual Analysis (Testing Linear Regression Assumptions)
    residuals = y_test - y_test_pred
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Residuals vs Fitted
    ax1.scatter(y_test_pred, residuals, color='#386cb0', alpha=0.7, edgecolors='black')
    ax1.axhline(0, color='red', linestyle='--', linewidth=1.5)
    ax1.set_title('Residuals vs. Fitted Values (Homoscedasticity Check)')
    ax1.set_xlabel('Fitted / Predicted Values')
    ax1.set_ylabel('Residuals (Actual - Predicted)')

    # Residual Distribution
    sns.histplot(residuals, kde=True, color='#7fc97f', edgecolor='black', ax=ax2)
    ax2.axvline(0, color='red', linestyle='--', linewidth=1.5)
    ax2.set_title('Distribution of Residuals (Normality Check)')
    ax2.set_xlabel('Residual Error')
    ax2.set_ylabel('Frequency')

    plt.suptitle('Statistical Diagnostic: Regression Residual Assumptions', fontsize=13, weight='bold')
    plt.tight_layout()
    resid_path = os.path.join(output_dir, 'residual_analysis.png')
    plt.savefig(resid_path, dpi=300)
    plt.close()
    print(f"Saved: {resid_path}")

    # Chart 3: Confusion Matrix Heatmap
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted Fail (<10)', 'Predicted Pass (>=10)'],
                yticklabels=['Actual Fail (<10)', 'Actual Pass (>=10)'],
                annot_kws={'size': 14, 'weight': 'bold'}, ax=ax)
    ax.set_title(f'Confusion Matrix: Student Pass/Fail Prediction\nAccuracy: {acc*100:.1f}%, F1-Score: {f1:.3f}')
    ax.set_ylabel('True Student Status')
    ax.set_xlabel('Predicted Student Status')
    plt.tight_layout()
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved: {cm_path}")

    # Chart 4: Top Model Coefficients (Feature Importance)
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': lr.coef_
    }).sort_values(by='Coefficient', key=abs, ascending=False).head(12)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ['#2ca02c' if c > 0 else '#d62728' for c in coef_df['Coefficient']]
    sns.barplot(x='Coefficient', y='Feature', data=coef_df, palette=colors, ax=ax, edgecolor='black')
    ax.axvline(0, color='black', linestyle='-', linewidth=0.8)
    ax.set_title('Top Influential Factors in Student Performance (Linear Regression Coefficients)')
    ax.set_xlabel('Regression Coefficient Magnitude (Slope β)')
    ax.set_ylabel('Academic & Demographic Feature')
    plt.tight_layout()
    coef_path = os.path.join(output_dir, 'feature_coefficients.png')
    plt.savefig(coef_path, dpi=300)
    plt.close()
    print(f"Saved: {coef_path}")

    return model_bundle

if __name__ == '__main__':
    train_and_evaluate_models()
