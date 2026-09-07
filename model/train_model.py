"""
Career Path Navigator Model Training Script

This script implements a proper machine learning pipeline with:
- Train/test split (80/20)
- Multiple model comparison
- Comprehensive evaluation metrics
- Reproducible results with fixed random states
"""

import pathlib
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

try:
    # Works when imported as part of the `model` package (e.g. `python -m model.train_model`)
    from model.evaluation import (
        compare_models,
        compare_class_weights,
        print_evaluation_report,
        cross_validate_model,
        evaluate_baseline,
        evaluate_model,
        get_top_features_per_class,
    )
except ImportError:
    # Works when run directly as a script from inside the `model/` directory
    from evaluation import (
        compare_models,
        compare_class_weights,
        print_evaluation_report,
        cross_validate_model,
        evaluate_baseline,
        evaluate_model,
        get_top_features_per_class,
    )


RANDOM_STATE = 42


def _save_confusion_matrix_plot(conf_matrix, class_names, out_path):
    """Render and save a confusion matrix heatmap for demo/reporting purposes."""
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend, safe for headless training runs
    import matplotlib.pyplot as plt

    conf_matrix = np.asarray(conf_matrix)
    if conf_matrix.shape[0] != len(class_names) or conf_matrix.shape[1] != len(class_names):
        raise ValueError(
            f"confusion matrix shape {conf_matrix.shape} doesn't match "
            f"{len(class_names)} class_names -- refusing to plot with "
            f"mismatched labels rather than mislabel the axes."
        )

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(conf_matrix, cmap="Blues")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted Career")
    ax.set_ylabel("True Career")
    ax.set_title("Confusion Matrix (Test Set)")

    # Annotate each cell with its count
    thresh = conf_matrix.max() / 2.0 if conf_matrix.max() > 0 else 0
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            value = conf_matrix[i, j]
            ax.text(
                j, i, str(value),
                ha="center", va="center",
                color="white" if value > thresh else "black",
                fontsize=9,
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def load_data(csv_path: str) -> pd.DataFrame:
    """Load and clean dataset."""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["skills", "interests", "education", "experience", "career_category"])
    df = df.astype(
        {
            "skills": str,
            "interests": str,
            "education": str,
            "experience": str,
            "career_category": str,
        }
    )
    return df


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Keep skills/interests/education/experience as separate typed columns
    (not one concatenated string) so ColumnTransformer can apply TF-IDF,
    one-hot encoding, and scaling to each appropriately."""
    return pd.DataFrame(
        {
            "skills": df["skills"].str.lower(),
            "interests": df["interests"].str.lower(),
            "education": df["education"].str.lower(),
            "experience": pd.to_numeric(df["experience"], errors="coerce").fillna(0),
        }
    )


def create_model_pipelines():
    """Build the 3 comparison pipelines (default hyperparameters -- tuning
    happens separately via GridSearchCV)."""
    def make_preprocessor():
        return ColumnTransformer(
            transformers=[
                # TF-IDF on skills/interests (unigrams only -- skill lists are
                # unordered keywords, not sentences, so bigrams would just
                # capture arbitrary token-adjacency; see METHODOLOGY.md item 11).
                ("skills_tfidf", TfidfVectorizer(ngram_range=(1, 1), max_features=1000, min_df=1), "skills"),
                ("interests_tfidf", TfidfVectorizer(ngram_range=(1, 1), max_features=1000, min_df=1), "interests"),
                # One-Hot Encoding: education has no natural numeric order.
                ("education_ohe", OneHotEncoder(handle_unknown="ignore"), ["education"]),
                # StandardScaler: puts numeric experience on the same scale as the other features.
                ("experience_num", StandardScaler(), ["experience"]),
            ]
        )

    pipelines = {
        "Logistic Regression": Pipeline(
            [
                ("preprocessor", make_preprocessor()),
                ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("preprocessor", make_preprocessor()),
                ("clf", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)),
            ]
        ),
        "Linear SVM": Pipeline(
            [
                ("preprocessor", make_preprocessor()),
                ("clf", SVC(kernel="linear", random_state=RANDOM_STATE, probability=True)),
            ]
        ),
    }
    return pipelines


def make_class_weight_variant(model_name: str, class_weight):
    """Build an unfitted pipeline for `model_name` with a given class_weight
    (None or "balanced") -- used only by the optional class-weight experiment."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("skills_tfidf", TfidfVectorizer(ngram_range=(1, 1), max_features=1000, min_df=1), "skills"),
            ("interests_tfidf", TfidfVectorizer(ngram_range=(1, 1), max_features=1000, min_df=1), "interests"),
            ("education_ohe", OneHotEncoder(handle_unknown="ignore"), ["education"]),
            ("experience_num", StandardScaler(), ["experience"]),
        ]
    )
    if model_name == "Logistic Regression":
        clf = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, class_weight=class_weight)
    elif model_name == "Linear SVM":
        clf = SVC(kernel="linear", random_state=RANDOM_STATE, probability=True, class_weight=class_weight)
    elif model_name == "Random Forest":
        clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, class_weight=class_weight)
    else:
        raise ValueError(f"Unknown model_name for class-weight experiment: {model_name}")
    return Pipeline([("preprocessor", preprocessor), ("clf", clf)])


def run_hyperparameter_tuning(X_train, y_train, cv=5):
    """
    Simple hyperparameter tuning demonstration using GridSearchCV.

    We tune exactly one hyperparameter (C, the regularization strength) for
    exactly one model (Logistic Regression). This is intentionally small in
    scope -- the goal is to demonstrate understanding of *why* and *how*
    hyperparameter tuning works, not to build an exhaustive search.

    What C controls: Logistic Regression tries to balance two things --
    fitting the training data well, and keeping its coefficients small
    (regularization) to avoid overfitting. C is the *inverse* of
    regularization strength: a SMALL C means MORE regularization (simpler,
    more conservative model), a LARGE C means LESS regularization (fits the
    training data more closely). GridSearchCV tries each candidate value,
    scores it with cross-validation, and reports which one generalizes best.

    Returns:
        dict with the parameter grid tested, per-value CV scores, and the
        best C found -- all computed from an actual GridSearchCV run, not
        hardcoded.
    """
    from sklearn.model_selection import GridSearchCV

    tuning_pipeline = Pipeline(
        [
            ("preprocessor", create_model_pipelines()["Logistic Regression"].named_steps["preprocessor"]),
            ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]
    )

    # Small, deliberately simple grid -- just the one hyperparameter that
    # matters most for Logistic Regression's bias/variance tradeoff.
    param_grid = {"clf__C": [0.1, 1, 5, 10, 20]}

    grid_search = GridSearchCV(
        tuning_pipeline,
        param_grid=param_grid,
        cv=cv,  # 5-fold (stratified automatically, since clf is a classifier)
        scoring="accuracy",
        n_jobs=None,
    )
    grid_search.fit(X_train, y_train)

    results_per_c = [
        {"C": params["clf__C"], "mean_cv_accuracy": float(mean_score), "std_cv_accuracy": float(std_score)}
        for params, mean_score, std_score in zip(
            grid_search.cv_results_["params"],
            grid_search.cv_results_["mean_test_score"],
            grid_search.cv_results_["std_test_score"],
        )
    ]

    return {
        "param_grid": param_grid,
        "results_per_value": results_per_c,
        "best_C": grid_search.best_params_["clf__C"],
        "best_cv_accuracy": float(grid_search.best_score_),
        "best_estimator": grid_search.best_estimator_,
    }


def _save_learning_curve_plot(estimator, X, y, out_path, cv=5):
    """
    Generate and save a learning curve: train vs. cross-validation score as
    the size of the training set grows. This is the standard plot for
    diagnosing whether a model would benefit from more data (still climbing)
    or has plateaued (converged), and for spotting overfitting (large gap
    between train and validation curves).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.model_selection import learning_curve

    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y,
        cv=cv,
        scoring="accuracy",
        train_sizes=np.linspace(0.2, 1.0, 6),
        random_state=RANDOM_STATE,
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(train_sizes, train_mean, "o-", color="#0E63F6", label="Training score")
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                     alpha=0.15, color="#0E63F6")
    ax.plot(train_sizes, val_mean, "o-", color="#0EA66B", label="Cross-validation score")
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                     alpha=0.15, color="#0EA66B")

    ax.set_xlabel("Training set size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Learning Curve")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def train(csv_path: str, model_path: str, enc_path: str, eval_path: str = None):
    """
    Train and evaluate models with proper train/test split.
    
    Args:
        csv_path: Path to training data CSV
        model_path: Path to save trained model
        enc_path: Path to save label encoder
        eval_path: Path to save evaluation results (optional)
    """
    print("=" * 70)
    print("CAREER PATH NAVIGATOR - MODEL TRAINING PIPELINE")
    print("=" * 70)

    # Load data
    print("\n1. Loading data...")
    df = load_data(csv_path)
    print(f"   Dataset size: {len(df)} records")
    print(f"   Unique careers: {df['career_category'].nunique()}")
    print(f"   Career distribution:\n{df['career_category'].value_counts()}")

    # Prepare features and labels. Skills/interests/education/experience are
    # kept as separate structured columns (see build_feature_frame) instead
    # of being concatenated into one text blob -- this lets the model treat
    # education/experience as categorical/numeric signals in their own
    # right, which produces sharper, more genuinely-justified confidence
    # scores on ambiguous input than lumping everything into text tokens.
    X_features = build_feature_frame(df)
    y = df["career_category"]

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Train/test split (80/20 with stratification)
    print("\n2. Splitting data (80/20 stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_encoded, test_size=0.2, random_state=RANDOM_STATE, stratify=y_encoded
    )
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")

    # Create model pipelines
    print("\n3. Creating model pipelines...")
    models = create_model_pipelines()
    for model_name in models.keys():
        print(f"   - {model_name}")

    # Baseline comparison: what accuracy would a trivial "always predict the
    # most common class" strategy get? This gives an honest reference point
    # for whether the trained model is actually learning signal.
    print("\n4a. Computing majority-class baseline...")
    baseline_results = evaluate_baseline(y_train, y_test)
    print(f"   Majority-class baseline accuracy: {baseline_results['baseline_accuracy']:.4f}")
    print(f"   (Best model accuracy will be compared against this below)")

    # Compare models
    print("\n4. Training and evaluating models...")
    comparison_df, detailed_results = compare_models(
        models, X_train, y_train, X_test, y_test, label_encoder, cv=5
    )

    # Display comparison
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print(comparison_df.to_string(index=False))
    print("=" * 70)

    # Select best model using CROSS-VALIDATION ONLY, never the test set.
    # Both the primary key (CV Mean) and the tie-break (CV Std, lower is
    # more consistent) come from cross_validate_model(), which runs
    # entirely on X_train/y_train. Test accuracy is reported afterwards for
    # transparency but has zero influence on which model gets selected.
    sorted_df = comparison_df.sort_values(by=["CV Mean", "CV Std"], ascending=[False, True])
    best_idx = sorted_df.index[0]
    best_model_name = comparison_df.loc[best_idx, "Model"]
    best_cv_mean = comparison_df.loc[best_idx, "CV Mean"]
    best_accuracy = comparison_df.loc[best_idx, "Accuracy"]

    print(f"\n5. Selected Model (by CV Mean, training data only): {best_model_name}")
    print(f"   CV Mean: {best_cv_mean:.4f}")
    print(f"   Test Accuracy (reported after selection, not used to select): {best_accuracy:.4f}")

    # Train final model on best model
    final_model = models[best_model_name]
    final_model.fit(X_train, y_train)
    y_pred = final_model.predict(X_test)

    best_metrics = detailed_results[best_model_name]["test_metrics"]
    cv_metrics = detailed_results[best_model_name]["cv_metrics"]

    # ------------------------------------------------------------------
    # 6. Hyperparameter tuning (simple demonstration): GridSearchCV over a
    # small grid of C values for Logistic Regression only. This runs
    # regardless of which model won the comparison above, so its results
    # are always genuine and reproducible -- but the tuned model is only
    # actually deployed if Logistic Regression is the model that won.
    # ------------------------------------------------------------------
    print("\n6. Hyperparameter tuning (GridSearchCV on Logistic Regression's C)...")
    tuning_results = run_hyperparameter_tuning(X_train, y_train, cv=5)
    print(f"   Tested C values: {tuning_results['param_grid']['clf__C']}")
    for row in tuning_results["results_per_value"]:
        print(f"   C={row['C']:>5}: mean CV accuracy = {row['mean_cv_accuracy']:.4f} (+/- {row['std_cv_accuracy']:.4f})")
    print(f"   Best C found: {tuning_results['best_C']} (CV accuracy: {tuning_results['best_cv_accuracy']:.4f})")

    tuning_applied = False
    if best_model_name == "Logistic Regression":
        # The deployed model IS Logistic Regression, so use the tuned
        # version found by GridSearchCV instead of the default-C one, and
        # recompute its test/CV metrics from scratch -- reusing the
        # default-C model's metrics here would silently report numbers
        # that don't match what's actually being saved/deployed.
        final_model = tuning_results["best_estimator"]
        y_pred = final_model.predict(X_test)
        best_metrics = evaluate_model(final_model, X_test, y_test, y_pred, best_model_name)
        cv_metrics = cross_validate_model(final_model, X_train, y_train, cv=5)
        tuning_applied = True
        print(f"   Applied: final Logistic Regression model now uses C={tuning_results['best_C']}")
        print(f"   Re-evaluated on test set with tuned model: accuracy={best_metrics['accuracy']:.4f}")
    else:
        print(f"   Not applied: {best_model_name} (not Logistic Regression) was selected as the "
              f"deployed model, so this tuning result is reported for demonstration only.")

    # ------------------------------------------------------------------
    # 6b. Class-weight experiment (kept as a secondary, optional check --
    # not part of the main model-selection story). class_weight="balanced"
    # vs. the default (None), for all three models, on the same split.
    # Saved to evaluation_results.json for the "Advanced / optional
    # analysis" section on /evaluate, but not printed in detail here.
    # ------------------------------------------------------------------
    class_weight_factories = {
        name: (lambda cw, _name=name: make_class_weight_variant(_name, cw))
        for name in ("Logistic Regression", "Linear SVM", "Random Forest")
    }
    class_weight_results = compare_class_weights(
        class_weight_factories, X_train, y_train, X_test, y_test, label_encoder, cv=5
    )

    # Print detailed evaluation
    print_evaluation_report(best_metrics, label_encoder)

    print(f"\nBaseline Comparison:")
    print(f"   Majority-class baseline accuracy: {baseline_results['baseline_accuracy']:.4f}")
    print(f"   {best_model_name} test accuracy:       {best_metrics['accuracy']:.4f}")
    print(f"   Lift over baseline:               +{best_metrics['accuracy'] - baseline_results['baseline_accuracy']:.4f}")

    print(f"\nCross-Validation Results:")
    print(f"   Mean Accuracy: {cv_metrics['cv_mean']:.4f}")
    print(f"   Std Dev:       {cv_metrics['cv_std']:.4f}")
    print(f"   Fold Scores:   {[f'{s:.4f}' for s in cv_metrics['cv_scores']]}")

    # Save model and encoder
    print("\n7. Saving model and encoder...")
    pathlib.Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, model_path)
    joblib.dump(label_encoder, enc_path)
    print(f"   Model saved to: {model_path}")
    print(f"   Encoder saved to: {enc_path}")

    # Save evaluation results
    if eval_path:
        careers_sorted = sorted(label_encoder.classes_.tolist())

        # Per-class metrics table (precision/recall/f1/support), keyed by career name.
        # sklearn's classification_report keys per-class entries by the *string* of the
        # encoded label (e.g. "0", "1", ...), so we translate those back to career names.
        raw_report = best_metrics["classification_report"]
        per_class_report = {}
        for key, values in raw_report.items():
            if key in ("accuracy", "macro avg", "weighted avg"):
                continue
            career_name = label_encoder.inverse_transform([int(key)])[0]
            per_class_report[career_name] = {
                "precision": float(values["precision"]),
                "recall": float(values["recall"]),
                "f1_score": float(values["f1-score"]),
                "support": int(values["support"]),
            }

        top_features = get_top_features_per_class(final_model, label_encoder, top_n=10)

        eval_results = {
            "best_model": best_model_name,
            "test_accuracy": float(best_metrics["accuracy"]),
            "test_precision_weighted": float(best_metrics["precision_weighted"]),
            "test_recall_weighted": float(best_metrics["recall_weighted"]),
            "test_f1_weighted": float(best_metrics["f1_weighted"]),
            "test_precision_macro": float(best_metrics["precision_macro"]),
            "test_recall_macro": float(best_metrics["recall_macro"]),
            "test_f1_macro": float(best_metrics["f1_macro"]),
            "cv_mean": float(cv_metrics["cv_mean"]),
            "cv_std": float(cv_metrics["cv_std"]),
            "cv_scores": [float(s) for s in cv_metrics["cv_scores"]],
            "model_comparison": comparison_df.to_dict(orient="records"),
            "dataset_size": len(df),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "n_careers": int(df["career_category"].nunique()),
            "careers": careers_sorted,
            "confusion_matrix": best_metrics["confusion_matrix"].tolist(),
            "confusion_matrix_labels": label_encoder.inverse_transform(
                sorted(set(y_test) | set(y_pred))
            ).tolist(),
            "per_class_report": per_class_report,
            "baseline_accuracy": float(baseline_results["baseline_accuracy"]),
            "baseline_f1_weighted": float(baseline_results["baseline_f1_weighted"]),
            "accuracy_lift_over_baseline": float(
                best_metrics["accuracy"] - baseline_results["baseline_accuracy"]
            ),
            "top_features_per_class": top_features,
            "hyperparameter_tuning": {
                "method": "GridSearchCV",
                "model_tuned": "Logistic Regression",
                "param_grid": tuning_results["param_grid"],
                "results_per_value": tuning_results["results_per_value"],
                "best_C": tuning_results["best_C"],
                "best_cv_accuracy": tuning_results["best_cv_accuracy"],
                "applied_to_deployed_model": tuning_applied,
            },
            "class_weight_comparison": class_weight_results,
        }
        pathlib.Path(eval_path).parent.mkdir(parents=True, exist_ok=True)
        with open(eval_path, "w") as f:
            json.dump(eval_results, f, indent=2)
        print(f"   Evaluation saved to: {eval_path}")

        # Save a confusion matrix heatmap image for the demo/report. Use the
        # labels actually present in the test set/predictions (already
        # computed above as confusion_matrix_labels), not the full career
        # list -- if a class ever had zero test examples, the matrix would
        # have fewer rows/cols than careers_sorted, and mismatched tick
        # labels would silently mislabel the plot.
        try:
            _save_confusion_matrix_plot(
                best_metrics["confusion_matrix"],
                eval_results["confusion_matrix_labels"],
                pathlib.Path(eval_path).parent.parent / "static" / "confusion_matrix.png",
            )
        except Exception as plot_err:  # pragma: no cover - plotting is best-effort
            print(f"   (Skipped confusion matrix plot: {plot_err})")

        # Save a learning curve plot (train vs CV score as data size grows).
        # Uses final_model (the actually-deployed model, including the
        # tuned hyperparameters if tuning was applied above) rather than
        # the pre-tuning models[best_model_name], so the plot matches
        # what's really being shipped.
        try:
            _save_learning_curve_plot(
                final_model,
                X_train, y_train,
                pathlib.Path(eval_path).parent.parent / "static" / "learning_curve.png",
                cv=5,
            )
        except Exception as plot_err:  # pragma: no cover - plotting is best-effort
            print(f"   (Skipped learning curve plot: {plot_err})")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    return final_model, label_encoder


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train Career Path Navigator model with proper evaluation"
    )
    parser.add_argument(
        "--data", default="data/jobs_sample.csv", help="Path to the CSV training data"
    )
    parser.add_argument(
        "--model", default="model/career_model.joblib", help="Where to save the trained model"
    )
    parser.add_argument(
        "--encoder", default="model/label_encoder.joblib", help="Where to save the label encoder"
    )
    parser.add_argument(
        "--eval", default="model/evaluation_results.json", help="Where to save evaluation results"
    )

    args = parser.parse_args()
    train(args.data, args.model, args.encoder, args.eval)
