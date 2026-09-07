"""
Error analysis for the deployed model.

Re-creates the same train/test split used in train_model.py, loads the
saved model, and reports misclassified test rows (with confidence in both
the predicted and true class) plus the lowest-confidence predictions
overall. Nothing hardcoded -- run with: python -m model.error_analysis
"""

import json
import pathlib

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from model.train_model import build_feature_frame, load_data, RANDOM_STATE
except ImportError:
    from train_model import build_feature_frame, load_data, RANDOM_STATE

_MODEL_DIR = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _MODEL_DIR.parent


def run_error_analysis(
    csv_path=None,
    model_path=None,
    encoder_path=None,
    out_path=None,
    n_lowest_confidence=15,
):
    csv_path = pathlib.Path(csv_path) if csv_path else _PROJECT_ROOT / "data" / "jobs_sample.csv"
    model_path = pathlib.Path(model_path) if model_path else _MODEL_DIR / "career_model.joblib"
    encoder_path = pathlib.Path(encoder_path) if encoder_path else _MODEL_DIR / "label_encoder.joblib"
    out_path = pathlib.Path(out_path) if out_path else _MODEL_DIR / "error_analysis.json"

    df = load_data(csv_path)
    X_features = build_feature_frame(df)
    y = df["career_category"]

    pipeline = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)
    y_encoded = label_encoder.transform(y)

    # Same split params as train_model.py's train() -- must match, or this
    # wouldn't be the same held-out test set the model was evaluated on.
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_encoded, test_size=0.2, random_state=RANDOM_STATE, stratify=y_encoded
    )

    probs = pipeline.predict_proba(X_test)
    y_pred = np.argmax(probs, axis=1)

    test_skills = X_test["skills"].tolist()
    test_interests = X_test["interests"].tolist()
    test_education = X_test["education"].tolist()
    test_experience = X_test["experience"].tolist()

    misclassified = []
    for i in range(len(y_test)):
        if y_pred[i] != y_test[i]:
            true_career = label_encoder.inverse_transform([y_test[i]])[0]
            pred_career = label_encoder.inverse_transform([y_pred[i]])[0]
            misclassified.append(
                {
                    "skills": test_skills[i],
                    "interests": test_interests[i],
                    "education": test_education[i],
                    "experience": test_experience[i],
                    "true_career": true_career,
                    "predicted_career": pred_career,
                    "confidence_in_predicted": float(probs[i][y_pred[i]]),
                    "confidence_in_true_career": float(probs[i][y_test[i]]),
                }
            )

    # Lowest-confidence predictions overall, correct or not -- useful even
    # when there are very few outright errors.
    top_confidence = probs[np.arange(len(y_test)), y_pred]
    lowest_conf_idx = np.argsort(top_confidence)[:n_lowest_confidence]
    lowest_confidence = []
    for i in lowest_conf_idx:
        true_career = label_encoder.inverse_transform([y_test[i]])[0]
        pred_career = label_encoder.inverse_transform([y_pred[i]])[0]
        lowest_confidence.append(
            {
                "skills": test_skills[i],
                "interests": test_interests[i],
                "education": test_education[i],
                "experience": test_experience[i],
                "true_career": true_career,
                "predicted_career": pred_career,
                "correct": bool(y_pred[i] == y_test[i]),
                "confidence": float(top_confidence[i]),
            }
        )

    # Which (true, predicted) career pairs get confused, and how often.
    confusion_pairs = {}
    for row in misclassified:
        key = f"{row['true_career']} -> {row['predicted_career']}"
        confusion_pairs[key] = confusion_pairs.get(key, 0) + 1
    confusion_pairs = dict(sorted(confusion_pairs.items(), key=lambda kv: -kv[1]))

    # Check whether the minority class has a disproportionate error rate.
    test_class_counts = pd.Series(y_test).value_counts()
    error_class_counts = pd.Series([label_encoder.transform([m["true_career"]])[0] for m in misclassified]).value_counts() \
        if misclassified else pd.Series(dtype=int)
    minority_encoded = int(test_class_counts.idxmin())
    minority_class_name = label_encoder.inverse_transform([minority_encoded])[0]
    minority_test_count = int(test_class_counts.min())
    minority_error_count = int(error_class_counts.get(minority_encoded, 0))

    result = {
        "test_set_size": len(y_test),
        "total_errors": len(misclassified),
        "error_rate": len(misclassified) / len(y_test),
        "misclassified_examples": misclassified,
        "confusion_pairs": confusion_pairs,
        "lowest_confidence_predictions": lowest_confidence,
        "minority_class_check": {
            "minority_class": minority_class_name,
            "minority_class_test_count": minority_test_count,
            "minority_class_errors": minority_error_count,
            "minority_class_error_rate": (
                minority_error_count / minority_test_count if minority_test_count else None
            ),
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print("=" * 70)
    print("ERROR ANALYSIS")
    print("=" * 70)
    print(f"Test set size: {result['test_set_size']}")
    print(f"Total errors: {result['total_errors']} ({result['error_rate']*100:.1f}%)")
    print("\nConfusion pairs (true -> predicted):")
    for pair, count in confusion_pairs.items():
        print(f"  {pair}: {count}")
    print(f"\nMinority class ({minority_class_name}, {minority_test_count} test samples): "
          f"{minority_error_count} errors "
          f"({result['minority_class_check']['minority_class_error_rate']*100:.1f}% error rate)"
          if minority_test_count else "")
    print(f"\nSaved full analysis to: {out_path}")

    return result


if __name__ == "__main__":
    run_error_analysis()
