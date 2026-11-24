"""
Data Science Fundamentals and Statistical Intuition
Project-1: Predictive Analytics for Student Performance in Exams
Module: Deployment - Interactive Student Performance Prediction System
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import joblib
import pandas as pd
import numpy as np

MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'linear_regression_model.joblib'))

class StudentPerformancePredictor:
    def __init__(self, model_path=MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Train the model first using src/model.py.")
        bundle = joblib.load(model_path)
        self.model = bundle['model']
        self.feature_names = bundle['feature_names']
        self.metrics = bundle.get('metrics', {})

    def get_default_profile(self):
        """Returns a baseline student feature profile with medians/modes."""
        # Baseline profile based on typical student in dataset
        profile = {feat: 0.0 for feat in self.feature_names}
        
        # Populate realistic defaults
        defaults = {
            'school': 1, 'sex': 0, 'age': 16, 'address': 1, 'famsize': 1, 'Pstatus': 1,
            'Medu': 3, 'Fedu': 3, 'traveltime': 1, 'studytime': 2, 'failures': 0,
            'schoolsup': 0, 'famsup': 1, 'paid': 0, 'activities': 1, 'nursery': 1,
            'higher': 1, 'internet': 1, 'romantic': 0, 'famrel': 4, 'freetime': 3,
            'goout': 3, 'Dalc': 1, 'Walc': 1, 'health': 4, 'absences': 3,
            'G1': 11, 'G2': 11, 'prior_avg_grade': 11.0, 'support_index': 1
        }
        for k, v in defaults.items():
            if k in profile:
                profile[k] = v
        return profile

    def predict(self, student_input: dict):
        """
        Takes a dictionary of student features, populates missing defaults,
        and returns prediction metrics and academic advisory.
        """
        full_features = self.get_default_profile()
        full_features.update(student_input)

        # Ensure engineered features are calculated if primary variables provided
        if 'G1' in student_input or 'G2' in student_input:
            g1 = full_features.get('G1', 10)
            g2 = full_features.get('G2', 10)
            full_features['prior_avg_grade'] = (g1 + g2) / 2.0

        if any(k in student_input for k in ['schoolsup', 'famsup', 'paid']):
            full_features['support_index'] = (
                full_features.get('schoolsup', 0) +
                full_features.get('famsup', 0) +
                full_features.get('paid', 0)
            )

        df_input = pd.DataFrame([full_features])[self.feature_names]
        raw_pred = self.model.predict(df_input)[0]
        clipped_pred = float(np.clip(raw_pred, 0.0, 20.0))
        percentage = (clipped_pred / 20.0) * 100.0
        passed = clipped_pred >= 10.0

        # Qualitative Grade Category
        if clipped_pred >= 16.0:
            category = "Excellent (Distinction)"
            risk_level = "Very Low Risk"
        elif clipped_pred >= 14.0:
            category = "Good Performance"
            risk_level = "Low Risk"
        elif clipped_pred >= 10.0:
            category = "Satisfactory (Pass)"
            risk_level = "Moderate Risk"
        elif clipped_pred >= 8.0:
            category = "Marginal / Needs Intervention"
            risk_level = "High Risk"
        else:
            category = "Critical Failure Risk"
            risk_level = "Critical Risk"

        # Actionable Recommendations
        recommendations = []
        studytime = full_features.get('studytime', 2)
        failures = full_features.get('failures', 0)
        absences = full_features.get('absences', 0)
        g2 = full_features.get('G2', 10)

        if studytime < 3:
            recommendations.append("Increase weekly study hours to at least 5-10 hours (Level 3 or 4).")
        if absences > 5:
            recommendations.append(f"High absenteeism detected ({absences} absences). Prioritize regular class attendance.")
        if failures > 0:
            recommendations.append(f"Student has {failures} past failure(s). Enroll in structured remedial support or tutoring.")
        if g2 < 10:
            recommendations.append("Recent period-2 exam grade was below passing threshold. Targeted concept review needed.")
        if not recommendations:
            recommendations.append("Current study habits and performance are strong. Maintain consistent revision.")

        return {
            'predicted_grade': round(clipped_pred, 2),
            'percentage': round(percentage, 1),
            'status': 'PASS' if passed else 'FAIL',
            'grade_tier': category,
            'risk_level': risk_level,
            'input_summary': {
                'studytime': studytime,
                'prior_g1': full_features.get('G1'),
                'prior_g2': full_features.get('G2'),
                'absences': absences,
                'failures': failures
            },
            'recommendations': recommendations
        }

def interactive_cli():
    print("=" * 65)
    print(" L&T EduTech: Student Exam Performance Prediction System ")
    print(" Powered by Multiple Linear Regression Machine Learning Model ")
    print("=" * 65)
    predictor = StudentPerformancePredictor()

    print("\nEnter student academic parameters:")
    try:
        g1 = float(input("Period 1 Grade G1 (0 to 20) [e.g. 12]: ") or 12)
        g2 = float(input("Period 2 Grade G2 (0 to 20) [e.g. 13]: ") or 13)
        print("Weekly Study Time: 1 (<2 hrs), 2 (2-5 hrs), 3 (5-10 hrs), 4 (>10 hrs)")
        studytime = int(input("Study Time (1-4) [e.g. 2]: ") or 2)
        absences = int(input("School Absences (0-50) [e.g. 2]: ") or 2)
        failures = int(input("Past Class Failures (0-3) [e.g. 0]: ") or 0)
        medu = int(input("Mother's Education (0-none, 4-higher) [e.g. 3]: ") or 3)
        fedu = int(input("Father's Education (0-none, 4-higher) [e.g. 3]: ") or 3)
        schoolsup = 1 if input("Extra School Support (yes/no) [no]: ").strip().lower() == 'yes' else 0
        higher = 1 if input("Aspirations for Higher Education (yes/no) [yes]: ").strip().lower() != 'no' else 0

        student_dict = {
            'G1': g1, 'G2': g2, 'studytime': studytime, 'absences': absences,
            'failures': failures, 'Medu': medu, 'Fedu': fedu,
            'schoolsup': schoolsup, 'higher': higher
        }

        result = predictor.predict(student_dict)

        print("\n" + "-" * 50)
        print(" PREDICTION RESULTS & ACADEMIC EVALUATION ")
        print("-" * 50)
        print(f"Predicted Final Exam Grade (G3): {result['predicted_grade']} / 20 ({result['percentage']}%)")
        print(f"Outcome Status:                 {result['status']}")
        print(f"Academic Tier:                  {result['grade_tier']}")
        print(f"Risk Assessment:                {result['risk_level']}")
        print("\nAdvisory & Actionable Recommendations:")
        for idx, rec in enumerate(result['recommendations'], 1):
            print(f"  {idx}. {rec}")
        print("-" * 50)

    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == '__main__':
    # Demo with test personas
    print("Testing Student Performance Prediction System Deployment...")
    predictor = StudentPerformancePredictor()
    
    test_students = [
        {
            'name': 'Student A (High Achiever)',
            'data': {'G1': 16, 'G2': 17, 'studytime': 4, 'absences': 1, 'failures': 0, 'higher': 1}
        },
        {
            'name': 'Student B (Average / Borderline)',
            'data': {'G1': 9, 'G2': 10, 'studytime': 2, 'absences': 6, 'failures': 1, 'higher': 1}
        },
        {
            'name': 'Student C (At-Risk / In Need of Intervention)',
            'data': {'G1': 5, 'G2': 6, 'studytime': 1, 'absences': 14, 'failures': 2, 'higher': 0}
        }
    ]

    for student in test_students:
        print("\n" + "="*55)
        print(f"Evaluating: {student['name']}")
        res = predictor.predict(student['data'])
        print(f"Input: G1={student['data']['G1']}, G2={student['data']['G2']}, StudyTime={student['data']['studytime']}, Absences={student['data']['absences']}, Failures={student['data']['failures']}")
        print(f"-> Predicted Grade: {res['predicted_grade']}/20 ({res['percentage']}%) | Status: {res['status']} | Tier: {res['grade_tier']}")
        print(f"-> Risk Level: {res['risk_level']}")
        print(f"-> Recommendations: {', '.join(res['recommendations'])}")
