"""
Exploratory Data Analysis (EDA) for Career Path Navigator dataset.

This script provides comprehensive data analysis, a data quality summary,
and saved chart images (class distribution, top skills, salary spread)
for use in reports/presentations.
"""

import pandas as pd
import numpy as np
import json
import pathlib

_MODEL_DIR = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _MODEL_DIR.parent
_DEFAULT_CSV = _PROJECT_ROOT / "data" / "jobs_sample.csv"
_DEFAULT_STATIC_DIR = _PROJECT_ROOT / "static"


def _inr(n) -> str:
    """Format a number as Indian Rupees with lakh/crore grouping, e.g. ₹8,50,000."""
    n = int(round(n))
    s = str(abs(n))
    if len(s) <= 3:
        grouped = s
    else:
        last_three = s[-3:]
        rest = s[:-3]
        parts = []
        while len(rest) > 2:
            parts.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            parts.insert(0, rest)
        grouped = ",".join(parts) + "," + last_three
    sign = "-" if n < 0 else ""
    return f"{sign}₹{grouped}"


def _save_eda_charts(df, career_dist, skill_counts, output_dir: pathlib.Path):
    """Generate and save class distribution, top-skills, and salary charts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Career class distribution (bar chart) -- shows whether classes are balanced.
    fig, ax = plt.subplots(figsize=(9, 5))
    career_dist.sort_values().plot(kind="barh", ax=ax, color="#0E63F6")
    ax.set_xlabel("Number of records")
    ax.set_title("Career Category Distribution")
    fig.tight_layout()
    fig.savefig(output_dir / "eda_career_distribution.png", dpi=150)
    plt.close(fig)

    # 2. Top skills overall (bar chart)
    fig, ax = plt.subplots(figsize=(9, 6))
    skill_counts.head(15).sort_values().plot(kind="barh", ax=ax, color="#0EA66B")
    ax.set_xlabel("Frequency across dataset")
    ax.set_title("Top 15 Most Common Skills")
    fig.tight_layout()
    fig.savefig(output_dir / "eda_top_skills.png", dpi=150)
    plt.close(fig)

    # 3. Salary spread by career (box plot) -- in INR (lakhs, for readable axis labels)
    fig, ax = plt.subplots(figsize=(10, 6))
    careers_sorted = sorted(df["career_category"].unique())
    data = [
        (df.loc[df["career_category"] == c, "salary_max"] / 100000).values
        for c in careers_sorted
    ]
    # matplotlib renamed boxplot's `labels` kwarg to `tick_labels` in 3.9 and
    # will drop the old name in 3.11. requirements.txt only guarantees
    # matplotlib>=3.7, so we can't assume either name is safe -- detect which
    # one this installed version actually accepts instead of hardcoding one.
    import inspect
    boxplot_params = inspect.signature(ax.boxplot).parameters
    label_kwarg = "tick_labels" if "tick_labels" in boxplot_params else "labels"
    ax.boxplot(data, vert=False, **{label_kwarg: careers_sorted})
    ax.set_xlabel("Maximum salary (₹ Lakhs Per Annum)")
    ax.set_title("Salary Spread by Career (LPA)")
    fig.tight_layout()
    fig.savefig(output_dir / "eda_salary_by_career.png", dpi=150)
    plt.close(fig)


    # 4. Education level distribution (bar chart) -- what's the mix of
    #    education levels across the whole dataset?
    fig, ax = plt.subplots(figsize=(8, 5))
    edu_dist = df["education"].value_counts()
    edu_dist.sort_values().plot(kind="barh", ax=ax, color="#8B5CF6")
    ax.set_xlabel("Number of records")
    ax.set_title("Education Level Distribution")
    fig.tight_layout()
    fig.savefig(output_dir / "eda_education_distribution.png", dpi=150)
    plt.close(fig)

    # 5. Experience distribution (histogram) -- how many years of
    #    experience do records in the dataset typically have?
    fig, ax = plt.subplots(figsize=(8, 5))
    exp_values = pd.to_numeric(df["experience"], errors="coerce").dropna()
    bins = range(int(exp_values.min()), int(exp_values.max()) + 2)
    ax.hist(exp_values, bins=bins, color="#F59E0B", edgecolor="white", align="left")
    ax.set_xlabel("Years of experience")
    ax.set_ylabel("Number of records")
    ax.set_title("Experience Distribution")
    ax.set_xticks(list(bins)[:-1])
    fig.tight_layout()
    fig.savefig(output_dir / "eda_experience_distribution.png", dpi=150)
    plt.close(fig)

    # 6. Most common interests overall (bar chart) -- same idea as top
    #    skills, but for the `interests` column.
    all_interests = []
    for interests_str in df["interests"]:
        all_interests.extend(i.strip().lower() for i in str(interests_str).split(",") if i.strip())
    interest_counts = pd.Series(all_interests).value_counts()

    fig, ax = plt.subplots(figsize=(9, 6))
    interest_counts.head(15).sort_values().plot(kind="barh", ax=ax, color="#EC4899")
    ax.set_xlabel("Frequency across dataset")
    ax.set_title("Top 15 Most Common Interests")
    fig.tight_layout()
    fig.savefig(output_dir / "eda_top_interests.png", dpi=150)
    plt.close(fig)

    # 7. Skills by career (small multiples) -- for each career, what are
    #    its top 5 most common skills? This answers "what actually defines
    #    each career in the dataset", one subplot per career, so it stays
    #    easy to read even with 9 categories instead of cramming everything
    #    into one dense grouped bar chart.
    careers_sorted = sorted(df["career_category"].unique())
    n_careers = len(careers_sorted)
    n_cols = 3
    n_rows = (n_careers + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 3.2 * n_rows))
    axes = axes.flatten()

    for i, career in enumerate(careers_sorted):
        career_df = df[df["career_category"] == career]
        career_skills = []
        for skills_str in career_df["skills"]:
            career_skills.extend(s.strip().lower() for s in str(skills_str).split(",") if s.strip())
        top5 = pd.Series(career_skills).value_counts().head(5)

        ax = axes[i]
        top5.sort_values().plot(kind="barh", ax=ax, color="#0E63F6")
        ax.set_title(career, fontsize=10)
        ax.set_xlabel("")
        ax.tick_params(axis="both", labelsize=8)

    # Hide any unused subplot slots (grid may have more cells than careers)
    for j in range(n_careers, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Top 5 Skills by Career", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_dir / "eda_skills_by_career.png", dpi=150)
    plt.close(fig)

    # 8. Interests by career (small multiples) -- same idea as skills-by-
    #    career above, but for the `interests` column. Answers "what does
    #    each career's audience say they're interested in, distinctly?"
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 3.2 * n_rows))
    axes = axes.flatten()

    for i, career in enumerate(careers_sorted):
        career_df = df[df["career_category"] == career]
        career_interests = []
        for interests_str in career_df["interests"]:
            career_interests.extend(x.strip().lower() for x in str(interests_str).split(",") if x.strip())
        top5 = pd.Series(career_interests).value_counts().head(5)

        ax = axes[i]
        top5.sort_values().plot(kind="barh", ax=ax, color="#EC4899")
        ax.set_title(career, fontsize=10)
        ax.set_xlabel("")
        ax.tick_params(axis="both", labelsize=8)

    for j in range(n_careers, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Top 5 Interests by Career", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_dir / "eda_interests_by_career.png", dpi=150)
    plt.close(fig)

    # 9. Experience vs Career -- does average experience differ meaningfully
    #    by career? (e.g. do senior-heavy careers like Product Manager skew
    #    higher than entry-heavy ones?) A simple bar of mean experience per
    #    career, sorted, answers this directly without needing a more
    #    complex plot.
    fig, ax = plt.subplots(figsize=(9, 6))
    exp_by_career = (
        df.assign(experience_num=pd.to_numeric(df["experience"], errors="coerce"))
        .groupby("career_category")["experience_num"]
        .mean()
        .sort_values()
    )
    exp_by_career.plot(kind="barh", ax=ax, color="#10B981")
    ax.set_xlabel("Mean years of experience")
    ax.set_title("Average Experience by Career")
    fig.tight_layout()
    fig.savefig(output_dir / "eda_experience_by_career.png", dpi=150)
    plt.close(fig)


def perform_eda(csv_path: str = None, output_dir: str = None, static_dir: str = None):
    """
    Perform comprehensive EDA on the dataset.

    Args:
        csv_path: Path to the dataset CSV (defaults to data/jobs_sample.csv
            relative to the project, regardless of current working directory)
        output_dir: Directory to save the eda_results.json summary
        static_dir: Directory to save chart images (for use in the app/reports)
    """
    csv_path = pathlib.Path(csv_path) if csv_path else _DEFAULT_CSV
    output_dir = pathlib.Path(output_dir) if output_dir else _MODEL_DIR
    static_dir = pathlib.Path(static_dir) if static_dir else _DEFAULT_STATIC_DIR

    print("=" * 70)
    print("EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 70)

    # Load data
    df = pd.read_csv(csv_path)

    # Basic statistics
    print("\n1. DATASET OVERVIEW")
    print("-" * 70)
    print(f"Total Records: {len(df)}")
    print(f"Total Columns: {len(df.columns)}")
    print(f"Data Types:\n{df.dtypes}")

    # ------------------------------------------------------------------
    # DATA CLEANING / DATA QUALITY CHECKS
    # This is a simple, explainable data-cleaning pass -- the kind of
    # checks any dataset should get before modeling: missing values,
    # duplicate rows, empty text fields, and values outside the expected
    # range. Nothing fancy -- just pandas boolean checks -- but each one
    # answers a real "is this dataset actually clean?" question.
    # ------------------------------------------------------------------
    print("\n1b. DATA CLEANING CHECKS")
    print("-" * 70)

    missing_counts = df.isnull().sum()
    n_missing_total = int(missing_counts.sum())
    print(f"Missing values per column:\n{missing_counts}")

    n_duplicates = int(df.duplicated().sum())
    print(f"Duplicate rows: {n_duplicates}")

    # Empty skills/interests: a row could technically have a non-null but
    # blank/whitespace-only string, which .isnull() would miss.
    empty_skills = df["skills"].astype(str).str.strip().eq("").sum()
    empty_interests = df["interests"].astype(str).str.strip().eq("").sum()
    print(f"Empty 'skills' entries: {empty_skills}")
    print(f"Empty 'interests' entries: {empty_interests}")

    # Invalid value checks: things that would be nonsensical for this
    # dataset (negative/zero salary, salary_max below salary_min,
    # experience outside a sane 0-60 year range).
    invalid_salary_order = (df["salary_max"] < df["salary_min"]).sum()
    non_positive_salary = ((df["salary_min"] <= 0) | (df["salary_max"] <= 0)).sum()
    exp_numeric = pd.to_numeric(df["experience"], errors="coerce")
    invalid_experience = ((exp_numeric < 0) | (exp_numeric > 60) | exp_numeric.isna()).sum()
    print(f"Rows where salary_max < salary_min: {invalid_salary_order}")
    print(f"Rows with non-positive salary values: {non_positive_salary}")
    print(f"Rows with invalid/out-of-range experience: {invalid_experience}")

    data_cleaning_summary = {
        "missing_values_total": n_missing_total,
        "duplicate_rows": n_duplicates,
        "empty_skills_rows": int(empty_skills),
        "empty_interests_rows": int(empty_interests),
        "invalid_salary_order_rows": int(invalid_salary_order),
        "non_positive_salary_rows": int(non_positive_salary),
        "invalid_experience_rows": int(invalid_experience),
        "is_clean": bool(
            n_missing_total == 0
            and n_duplicates == 0
            and empty_skills == 0
            and empty_interests == 0
            and invalid_salary_order == 0
            and non_positive_salary == 0
            and invalid_experience == 0
        ),
    }
    print(f"\nOverall: {'✓ Dataset passed all data-cleaning checks' if data_cleaning_summary['is_clean'] else '✗ Issues found -- see above'}")

    # Career distribution
    print("\n2. CAREER CATEGORY DISTRIBUTION")
    print("-" * 70)
    career_dist = df["career_category"].value_counts()
    print(career_dist)
    print(f"\nUnique Careers: {career_dist.shape[0]}")
    print(f"Career Balance:")
    for career, count in career_dist.items():
        pct = count / len(df) * 100
        print(f"  {career}: {count} ({pct:.1f}%)")

    # Education distribution
    print("\n3. EDUCATION DISTRIBUTION")
    print("-" * 70)
    edu_dist = df["education"].value_counts()
    print(edu_dist)

    # Experience distribution
    print("\n4. EXPERIENCE DISTRIBUTION")
    print("-" * 70)
    exp_dist = df["experience"].value_counts().sort_index()
    print(exp_dist)

    # Salary statistics (Indian Rupees, annual CTC)
    print("\n5. SALARY STATISTICS (INR, Annual CTC)")
    print("-" * 70)
    print(f"Minimum Salary Range:")
    print(f"  Min: {_inr(df['salary_min'].min())}")
    print(f"  Max: {_inr(df['salary_min'].max())}")
    print(f"  Mean: {_inr(df['salary_min'].mean())}")
    print(f"  Median: {_inr(df['salary_min'].median())}")

    print(f"\nMaximum Salary Range:")
    print(f"  Min: {_inr(df['salary_max'].min())}")
    print(f"  Max: {_inr(df['salary_max'].max())}")
    print(f"  Mean: {_inr(df['salary_max'].mean())}")
    print(f"  Median: {_inr(df['salary_max'].median())}")

    print(f"\nSalary Range by Career:")
    salary_by_career = df.groupby("career_category").agg({
        "salary_min": ["min", "mean", "max"],
        "salary_max": ["min", "mean", "max"]
    }).round(0)
    print(salary_by_career)

    # Skills analysis
    print("\n6. SKILLS ANALYSIS")
    print("-" * 70)
    all_skills = []
    for skills_str in df["skills"]:
        skills = [s.strip().lower() for s in skills_str.split(",")]
        all_skills.extend(skills)

    skill_counts = pd.Series(all_skills).value_counts()
    print(f"Total Unique Skills: {len(skill_counts)}")
    print(f"\nTop 15 Most Common Skills:")
    print(skill_counts.head(15))

    # Interests analysis
    print("\n7. INTERESTS ANALYSIS")
    print("-" * 70)
    all_interests = []
    for interests_str in df["interests"]:
        interests = [i.strip().lower() for i in interests_str.split(",")]
        all_interests.extend(interests)

    interest_counts = pd.Series(all_interests).value_counts()
    print(f"Total Unique Interests: {len(interest_counts)}")
    print(f"\nTop 15 Most Common Interests:")
    print(interest_counts.head(15))

    # Career vs Skill relationships
    print("\n8. CAREER - SKILL ASSOCIATIONS")
    print("-" * 70)
    for career in df["career_category"].unique():
        career_df = df[df["career_category"] == career]
        career_skills = []
        for skills_str in career_df["skills"]:
            skills = [s.strip().lower() for s in skills_str.split(",")]
            career_skills.extend(skills)

        skill_freq = pd.Series(career_skills).value_counts()
        print(f"\n{career} (n={len(career_df)}):")
        print(f"  Top skills: {', '.join(skill_freq.head(5).index.tolist())}")

    # Data quality report -- computed dynamically from the actual dataset,
    # not hardcoded, so it stays accurate as the dataset grows.
    print("\n9. DATA QUALITY ASSESSMENT")
    print("-" * 70)
    class_counts = career_dist
    class_balance_ratio = class_counts.min() / class_counts.max()
    is_balanced = class_balance_ratio >= 0.9

    quality_notes = []
    quality_notes.append(("pass", f"No missing values in critical columns" if df.isnull().sum().sum() == 0 else "Missing values detected"))
    quality_notes.append(("pass" if df.duplicated().sum() == 0 else "warn", f"{df.duplicated().sum()} duplicate records"))
    quality_notes.append(("pass", f"{len(df)} valid records"))
    quality_notes.append((
        "pass" if is_balanced else "warn",
        f"Class balance ratio (min/max class size): {class_balance_ratio:.2f} "
        f"({'balanced' if is_balanced else 'imbalanced'})"
    ))
    quality_notes.append(("warn", "No geographical information (single implicit market assumption)"))
    quality_notes.append(("warn", "No temporal information (no posting-date/year data)"))
    quality_notes.append(("warn", "Dataset is synthetically generated, not sourced from real job postings"))

    for status, note in quality_notes:
        symbol = "✓" if status == "pass" else "✗"
        print(f"{symbol} {note}")

    # Save chart images
    print("\n10. GENERATING CHARTS")
    print("-" * 70)
    try:
        _save_eda_charts(df, career_dist, skill_counts, static_dir)
        print(f"Charts saved to: {static_dir}")
    except Exception as chart_err:  # pragma: no cover - plotting is best-effort
        print(f"(Skipped chart generation: {chart_err})")

    # Save results to JSON
    print("\n11. SAVING RESULTS")
    print("-" * 70)
    output_dir.mkdir(parents=True, exist_ok=True)

    eda_results = {
        "data_cleaning": data_cleaning_summary,
        "dataset": {
            "total_records": len(df),
            "total_columns": len(df.columns),
            "missing_values": df.isnull().sum().to_dict(),
            "duplicate_rows": int(df.duplicated().sum()),
        },
        "careers": {
            "unique_count": int(df["career_category"].nunique()),
            "distribution": career_dist.to_dict(),
            "class_balance_ratio": float(class_balance_ratio),
        },
        "education": {
            "unique_values": df["education"].unique().tolist(),
            "distribution": edu_dist.to_dict(),
        },
        "experience": {
            "distribution": {str(k): int(v) for k, v in exp_dist.to_dict().items()},
        },
        "salary_inr": {
            "salary_min": {
                "min": float(df["salary_min"].min()),
                "max": float(df["salary_min"].max()),
                "mean": float(df["salary_min"].mean()),
                "median": float(df["salary_min"].median()),
            },
            "salary_max": {
                "min": float(df["salary_max"].min()),
                "max": float(df["salary_max"].max()),
                "mean": float(df["salary_max"].mean()),
                "median": float(df["salary_max"].median()),
            },
            "currency": "INR",
        },
        "skills": {
            "unique_count": len(skill_counts),
            "top_15": skill_counts.head(15).to_dict(),
        },
        "interests": {
            "unique_count": len(interest_counts),
            "top_15": interest_counts.head(15).to_dict(),
        },
        "data_quality_notes": [{"status": s, "note": n} for s, n in quality_notes],
    }

    eda_file = output_dir / "eda_results.json"
    with open(eda_file, "w") as f:
        json.dump(eda_results, f, indent=2)
    print(f"EDA results saved to: {eda_file}")

    print("\n" + "=" * 70)
    print("EDA COMPLETE")
    print("=" * 70)

    return eda_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Perform EDA on Career Path Navigator dataset")
    parser.add_argument(
        "--data", default=None, help="Path to the CSV dataset (defaults to data/jobs_sample.csv)"
    )
    parser.add_argument(
        "--output", default=None, help="Directory to save eda_results.json (defaults to model/)"
    )
    parser.add_argument(
        "--static-dir", default=None, help="Directory to save chart images (defaults to static/)"
    )

    args = parser.parse_args()
    perform_eda(args.data, args.output, args.static_dir)
