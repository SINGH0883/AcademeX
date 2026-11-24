"""
Data Science Fundamentals and Statistical Intuition
Project-1: Predictive Analytics for Student Performance in Exams
Module: Exploratory Data Analysis & Statistical Visualization
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set aesthetic style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 11

def generate_eda_visualizations(raw_df_path='data/student-mat.csv', output_dir='figures'):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(raw_df_path, sep=';')

    print("Generating EDA visualizations...")

    # 1. Distribution of Target Variable (Final Grade G3)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df['G3'], bins=21, kde=True, color='#1f77b4', edgecolor='black', ax=ax)
    ax.axvline(df['G3'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean Grade: {df["G3"].mean():.2f}')
    ax.axvline(df['G3'].median(), color='green', linestyle=':', linewidth=2, label=f'Median Grade: {df["G3"].median():.2f}')
    ax.axvline(10, color='darkorange', linestyle='-', linewidth=1.5, label='Passing Threshold (10/20)')
    ax.set_title('Distribution of Final Exam Grades (G3) with Statistical Benchmarks')
    ax.set_xlabel('Final Exam Grade (Scale: 0 - 20)')
    ax.set_ylabel('Number of Students')
    ax.legend(frameon=True)
    plt.tight_layout()
    dist_path = os.path.join(output_dir, 'distribution_target.png')
    plt.savefig(dist_path, dpi=300)
    plt.close()
    print(f"Saved: {dist_path}")

    # 2. Correlation Heatmap of Key Academic & Behavioral Features
    key_vars = ['studytime', 'failures', 'absences', 'Medu', 'Fedu', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'G1', 'G2', 'G3']
    corr_df = df[key_vars].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(corr_df, annot=True, fmt='.2f', cmap=cmap, vmin=-0.5, vmax=1.0,
                square=True, linewidths=0.5, cbar_kws={'shrink': 0.8}, ax=ax)
    ax.set_title('Correlation Heatmap: Academic & Behavioral Predictors vs Grades')
    plt.tight_layout()
    corr_path = os.path.join(output_dir, 'correlation_heatmap.png')
    plt.savefig(corr_path, dpi=300)
    plt.close()
    print(f"Saved: {corr_path}")

    # 3. Weekly Study Time vs Final Exam Score (Boxplot & Swarm/Strip)
    study_labels = {1: '<2 hours', 2: '2 to 5 hours', 3: '5 to 10 hours', 4: '>10 hours'}
    df['studytime_label'] = df['studytime'].map(study_labels)

    fig, ax = plt.subplots(figsize=(8, 5))
    palette = ['#e6f2ff', '#bdd7ee', '#6baed6', '#2171b5']
    sns.boxplot(x='studytime_label', y='G3', hue='studytime_label', data=df, palette=palette, legend=False, ax=ax, width=0.5, boxprops=dict(edgecolor='black'))
    sns.stripplot(x='studytime_label', y='G3', data=df, color='#08306b', alpha=0.4, jitter=0.2, size=5, ax=ax)
    ax.set_title('Impact of Weekly Study Hours on Final Exam Performance (G3)')
    ax.set_xlabel('Weekly Study Hours Category')
    ax.set_ylabel('Final Exam Grade (G3)')
    plt.tight_layout()
    study_path = os.path.join(output_dir, 'studytime_vs_grade.png')
    plt.savefig(study_path, dpi=300)
    plt.close()
    print(f"Saved: {study_path}")

    # 4. Prior Grades (G1, G2) vs Final Grade (G3) with Linear Fit
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    sns.regplot(x='G1', y='G3', data=df, ax=ax1,
                scatter_kws={'alpha': 0.6, 'color': '#2b5c8f'},
                line_kws={'color': '#d95f02', 'linewidth': 2})
    ax1.set_title('Period 1 Grade (G1) vs Final Grade (G3)')
    ax1.set_xlabel('First Period Grade (G1)')
    ax1.set_ylabel('Final Grade (G3)')

    sns.regplot(x='G2', y='G3', data=df, ax=ax2,
                scatter_kws={'alpha': 0.6, 'color': '#2ca02c'},
                line_kws={'color': '#7570b3', 'linewidth': 2})
    ax2.set_title('Period 2 Grade (G2) vs Final Grade (G3)')
    ax2.set_xlabel('Second Period Grade (G2)')
    ax2.set_ylabel('Final Grade (G3)')

    plt.suptitle('Strong Linear Relationship Between Prior Academic Grades and Final Exam Scores', fontsize=13, weight='bold')
    plt.tight_layout()
    prior_path = os.path.join(output_dir, 'prior_grades_vs_final.png')
    plt.savefig(prior_path, dpi=300)
    plt.close()
    print(f"Saved: {prior_path}")

    # 5. Absences and Past Failures Impact on Final Grade
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Failures vs Grade
    sns.barplot(x='failures', y='G3', hue='failures', data=df, palette='Reds_r', legend=False, ax=ax1, edgecolor='black', capsize=0.1)
    ax1.set_title('Average Final Grade by Number of Past Class Failures')
    ax1.set_xlabel('Number of Past Failures')
    ax1.set_ylabel('Mean Final Grade (G3)')

    # Absences vs Grade scatter
    sns.scatterplot(x='absences', y='G3', hue='failures', palette='viridis', data=df, s=60, ax=ax2)
    ax2.set_title('Absences vs Final Grade (Grouped by Past Failures)')
    ax2.set_xlabel('Number of School Absences')
    ax2.set_ylabel('Final Grade (G3)')
    ax2.legend(title='Past Failures')

    plt.suptitle('Risk Factors: Past Academic Failures & School Absenteeism', fontsize=13, weight='bold')
    plt.tight_layout()
    risk_path = os.path.join(output_dir, 'absences_failures_effect.png')
    plt.savefig(risk_path, dpi=300)
    plt.close()
    print(f"Saved: {risk_path}")

    print("All EDA charts generated successfully!")

if __name__ == '__main__':
    generate_eda_visualizations()
