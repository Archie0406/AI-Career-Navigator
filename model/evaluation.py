"""
Model evaluation utilities for the Career Path Navigator.

This module provides comprehensive evaluation functions for classification models,
including metrics calculation, model comparison, and reporting.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import cross_val_score


def evaluate_baseline(y_train, y_test):
    """
    Compute a trivial majority-class baseline: always predict the most
    frequent class seen in training. This gives an honest reference point
    for "is the model actually learning anything, or just exploiting class
    imbalance?" -- a very common data science interview question.

    Args:
        y_train: Training labels (encoded)
        y_test: Test labels (encoded)

    Returns:
        dict with baseline accuracy and the predicted majority class
    """
    from sklearn.dummy import DummyClassifier

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(np.zeros((len(y_train), 1)), y_train)
    y_pred_baseline = dummy.predict(np.zeros((len(y_test), 1)))

    baseline_accuracy = accuracy_score(y_test, y_pred_baseline)
    baseline_f1_weighted = f1_score(y_test, y_pred_baseline, average="weighted", zero_division=0)

    return {
        "baseline_accuracy": baseline_accuracy,
        "baseline_f1_weighted": baseline_f1_weighted,
        "majority_class_encoded": int(dummy.classes_[np.argmax(dummy.class_prior_)]) if hasattr(dummy, "class_prior_") else None,
    }


def get_top_features_per_class(pipeline, label_encoder, top_n: int = 10):
    """
    Extract the top-weighted features driving each class's predictions, for
    linear models that expose a proper one-row-per-class coefficient matrix
    (e.g. Logistic Regression). This is the interpretability check that
    turns "the model works" into "here's specifically what the model
    learned." Feature names come from the pipeline's ColumnTransformer
    ('preprocessor' step), which combines skills/interests TF-IDF terms,
    one-hot education categories, and the numeric experience feature into a
    single named feature space via get_feature_names_out().

    Note on Linear SVM: sklearn's SVC(kernel="linear") computes `coef_`
    internally as one-vs-one (shape (n_classes * (n_classes - 1) / 2,
    n_features) for n_classes > 2), not one row per class. That matrix isn't
    directly interpretable as "top terms for class X" without extra
    aggregation across the pairwise classifiers, so we deliberately skip it
    rather than mislabel pairwise coefficients as per-class ones. `coef_` can
    also be a sparse matrix when the input was sparse (as TF-IDF output is),
    so we densify before doing any indexing/arithmetic on it.

    Args:
        pipeline: A fitted sklearn Pipeline with 'preprocessor' and 'clf' steps
        label_encoder: LabelEncoder used to decode class indices to names
        top_n: How many top terms to keep per class

    Returns:
        dict mapping career name -> list of top terms, or None if the final
        classifier doesn't expose a usable one-row-per-class coefficient
        matrix (e.g. Random Forest, or a one-vs-one linear SVM).
    """
    clf = pipeline.named_steps.get("clf")
    preprocessor = pipeline.named_steps.get("preprocessor")

    if clf is None or preprocessor is None or not hasattr(clf, "coef_"):
        return None

    feature_names = preprocessor.get_feature_names_out()
    coefs = clf.coef_  # ideally shape: (n_classes, n_features) for multiclass linear models

    # coef_ may be a scipy sparse matrix (e.g. for SVC fit on sparse TF-IDF
    # input) -- densify so downstream indexing/argsort/float() all behave
    # like plain numpy arrays instead of sparse matrix objects.
    if hasattr(coefs, "toarray"):
        coefs = coefs.toarray()
    coefs = np.asarray(coefs)

    n_classes = len(label_encoder.classes_)

    if coefs.ndim != 2 or coefs.shape[0] != n_classes:
        # Not a one-row-per-class matrix (e.g. a one-vs-one linear SVM, or a
        # binary classifier's single-row coef_). Extracting per-class terms
        # here would either crash on unseen label codes or silently mislabel
        # pairwise coefficients as per-class ones -- skip instead.
        return None

    top_features = {}
    for class_idx in range(coefs.shape[0]):
        class_name = label_encoder.inverse_transform([class_idx])[0]
        row = coefs[class_idx]
        top_indices = np.argsort(row)[::-1][:top_n]
        top_features[class_name] = [
            {"term": _prettify_feature_name(feature_names[i]), "weight": float(row[i])}
            for i in top_indices
        ]

    return top_features


def _prettify_feature_name(raw_name: str) -> str:
    """
    ColumnTransformer feature names look like "skills_tfidf__python" or
    "education_ohe__education_phd" -- readable enough for debugging, but
    noisy for a demo UI. This turns them into e.g. "skills: python" /
    "education: phd" / "experience" so the interpretability table on
    /evaluate reads cleanly.
    """
    prefix_map = {
        "skills_tfidf__": "skills: ",
        "interests_tfidf__": "interests: ",
        "education_ohe__education_": "education: ",
        "experience_num__experience": "experience (years)",
    }
    for raw_prefix, label in prefix_map.items():
        if raw_name.startswith(raw_prefix):
            remainder = raw_name[len(raw_prefix):]
            return f"{label}{remainder}" if remainder else label
    return raw_name


def evaluate_model(model, X_test, y_test, y_pred, model_name: str = "Model"):
    """
    Evaluate a trained model on test data.
    
    Args:
        model: The trained model
        X_test: Test features
        y_test: True test labels
        y_pred: Predicted labels
        model_name: Name of the model for reporting
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    accuracy = accuracy_score(y_test, y_pred)
    
    # Use weighted average for multiclass
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    
    # Also compute macro averages for reference
    precision_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Per-class report: precision/recall/f1/support for every career category.
    # This is what actually reveals which classes the model confuses, versus
    # a single aggregate accuracy number which can hide weak classes.
    class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    metrics = {
        "model_name": model_name,
        "accuracy": accuracy,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "confusion_matrix": conf_matrix,
        "classification_report": class_report,
    }
    
    return metrics


def cross_validate_model(model, X_train, y_train, cv: int = 5):
    """
    Perform cross-validation on training data.
    
    Args:
        model: The model to cross-validate
        X_train: Training features
        y_train: Training labels
        cv: Number of CV folds
    
    Returns:
        dict: Dictionary with CV mean and std
    """
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    
    return {
        "cv_scores": cv_scores,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
    }


def compare_models(models_dict, X_train, y_train, X_test, y_test, label_encoder, cv: int = 5):
    """
    Compare multiple models on train and test data.
    
    Args:
        models_dict: Dictionary of {model_name: model_pipeline}
        X_train: Training features
        y_train: Training labels (encoded)
        X_test: Test features
        y_test: Test labels (encoded)
        label_encoder: LabelEncoder for career categories
        cv: Number of CV folds
    
    Returns:
        tuple: (comparison_df, detailed_results_dict)
    """
    results = []
    detailed_results = {}
    
    for model_name, model_pipeline in models_dict.items():
        print(f"\n--- Evaluating {model_name} ---")
        
        # Train on training data
        model_pipeline.fit(X_train, y_train)
        
        # Predict on test data
        y_pred = model_pipeline.predict(X_test)
        
        # Get test metrics
        test_metrics = evaluate_model(model_pipeline, X_test, y_test, y_pred, model_name)
        
        # Get cross-validation metrics on training data
        cv_metrics = cross_validate_model(model_pipeline, X_train, y_train, cv=cv)
        
        # Combine results
        combined_metrics = {
            "Model": model_name,
            "Accuracy": test_metrics["accuracy"],
            "Precision (weighted)": test_metrics["precision_weighted"],
            "Recall (weighted)": test_metrics["recall_weighted"],
            "F1-Score (weighted)": test_metrics["f1_weighted"],
            # Macro columns included alongside weighted ones (not just
            # buried in per-model detailed metrics) because the dataset is
            # now imbalanced -- macro F1 in particular is the number that
            # would expose a model doing well on big classes but badly on
            # small ones, which weighted/accuracy alone would hide.
            "Precision (macro)": test_metrics["precision_macro"],
            "Recall (macro)": test_metrics["recall_macro"],
            "F1-Score (macro)": test_metrics["f1_macro"],
            "CV Mean": cv_metrics["cv_mean"],
            "CV Std": cv_metrics["cv_std"],
        }
        
        results.append(combined_metrics)
        
        # Store detailed results
        detailed_results[model_name] = {
            "test_metrics": test_metrics,
            "cv_metrics": cv_metrics,
            "y_pred": y_pred,
        }
        
        print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Test F1-Score (weighted): {test_metrics['f1_weighted']:.4f}")
        print(f"CV Mean: {cv_metrics['cv_mean']:.4f} (+/- {cv_metrics['cv_std']:.4f})")
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results)
    
    return comparison_df, detailed_results


def compare_class_weights(model_factories, X_train, y_train, X_test, y_test, label_encoder, cv: int = 5):
    """Optional experiment: train each model with class_weight=None vs.
    "balanced" on the same split, to check if reweighting helps minority
    classes. Not part of the main model-selection flow -- see /evaluate's
    "Advanced / optional analysis" section for the results."""
    # Minority class = fewest TRAINING samples (what weighting is meant to help).
    train_class_counts = pd.Series(y_train).value_counts()
    minority_encoded = int(train_class_counts.idxmin())
    minority_class_name = label_encoder.inverse_transform([minority_encoded])[0]

    results = {}
    for model_name, make_pipeline in model_factories.items():
        variant_results = {}
        for variant_key, class_weight_value in [
            ("class_weight_none", None),
            ("class_weight_balanced", "balanced"),
        ]:
            pipeline = make_pipeline(class_weight_value)
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            metrics = evaluate_model(pipeline, X_test, y_test, y_pred, model_name)
            cv_metrics = cross_validate_model(pipeline, X_train, y_train, cv=cv)

            per_class = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            minority_recall = per_class.get(str(minority_encoded), {}).get("recall", None)

            variant_results[variant_key] = {
                "accuracy": float(metrics["accuracy"]),
                "precision_macro": float(metrics["precision_macro"]),
                "recall_macro": float(metrics["recall_macro"]),
                "f1_macro": float(metrics["f1_macro"]),
                "f1_weighted": float(metrics["f1_weighted"]),
                "cv_mean": float(cv_metrics["cv_mean"]),
                "minority_class_recall": float(minority_recall) if minority_recall is not None else None,
            }
        results[model_name] = variant_results

    return {
        "minority_class_name": minority_class_name,
        "minority_class_train_count": int(train_class_counts.min()),
        "results": results,
    }


def print_evaluation_report(metrics, label_encoder):
    """
    Print a formatted evaluation report.
    
    Args:
        metrics: Dictionary from evaluate_model()
        label_encoder: LabelEncoder for interpretation
    """
    print(f"\n{'='*60}")
    print(f"EVALUATION REPORT: {metrics['model_name']}")
    print(f"{'='*60}")
    print(f"Accuracy:          {metrics['accuracy']:.4f}")
    print(f"Precision (weighted): {metrics['precision_weighted']:.4f}")
    print(f"Recall (weighted):    {metrics['recall_weighted']:.4f}")
    print(f"F1-Score (weighted):  {metrics['f1_weighted']:.4f}")
    print(f"\nPrecision (macro): {metrics['precision_macro']:.4f}")
    print(f"Recall (macro):    {metrics['recall_macro']:.4f}")
    print(f"F1-Score (macro):  {metrics['f1_macro']:.4f}")
    print(f"\nNote: Weighted averages give equal weight to each sample.")
    print(f"      Macro averages give equal weight to each class.")
    print(f"{'='*60}\n")
