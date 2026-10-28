"""Reproducible synthetic-data training for a teaching-only loan risk demo."""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

ROOT = Path(__file__).resolve().parent
FEATURES = ['annual_income', 'loan_amount', 'loan_term_months', 'credit_score', 'existing_debt', 'employment_years', 'late_payments_12m']


def make_data(n=3500, seed=42):
    rng = np.random.default_rng(seed)
    income = rng.integers(180000, 2500001, n)
    amount = rng.integers(50000, 3000001, n)
    term = rng.choice([12, 24, 36, 48, 60, 84], n)
    credit = rng.integers(300, 901, n)
    debt = rng.integers(0, 1100001, n)
    years = rng.integers(0, 31, n)
    late = rng.integers(0, 7, n)
    # Synthetic target based on an illustrative formula, NOT real approvals.
    burden = amount / np.maximum(income, 1)
    debt_ratio = debt / np.maximum(income, 1)
    logit = (-0.4 + (credit - 650) / 115 + (income - 600000) / 900000
             - 1.1 * burden - 1.3 * debt_ratio + 0.035 * years - 0.40 * late)
    probability = 1 / (1 + np.exp(-np.clip(logit, -30, 30)))
    approved = rng.binomial(1, probability)
    return pd.DataFrame(dict(annual_income=income, loan_amount=amount,
                             loan_term_months=term, credit_score=credit,
                             existing_debt=debt, employment_years=years,
                             late_payments_12m=late, approved=approved))


def train():
    df = make_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df['approved'], test_size=0.25, random_state=42, stratify=df['approved'])
    model = RandomForestClassifier(n_estimators=180, min_samples_leaf=8,
                                   random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = {'accuracy': float(accuracy_score(y_test, predictions)),
               'roc_auc': float(roc_auc_score(y_test, probabilities)),
               'classification_report': classification_report(y_test, predictions, output_dict=True)}
    artifact = {'model': model, 'features': FEATURES, 'metrics': metrics, 'synthetic': True}
    joblib.dump(artifact, ROOT / 'loan_model.joblib')
    return artifact


if __name__ == '__main__':
    result = train()
    print('Synthetic test accuracy:', round(result['metrics']['accuracy'], 3))
    print('Synthetic test ROC AUC:', round(result['metrics']['roc_auc'], 3))
