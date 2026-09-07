"""
Career Path Navigator - Flask Application

This is a Data Science web application that recommends career paths based on
skills, interests, education, and experience.

COMPONENT SEPARATION:
- ML PREDICTION: best-performing classifier (selected during training, see
  model/evaluation_results.json) using predict_proba()
- DATASET LOOKUP: Skills and salary information from training data
- RULE-BASED: Manually defined career roadmaps
"""

import os
import json
from datetime import datetime

import pandas as pd
from flask import (
    Flask,
    render_template,
    request,
)

from model.predict import CareerModel
from model.roadmaps import CAREER_ROADMAPS, get_roadmap


def create_app():
    app = Flask(__name__)

    # Paths
    base = os.path.dirname(__file__)
    app.config["DATA_PATH"] = os.path.join(base, "data", "jobs_sample.csv")
    app.config["EVAL_PATH"] = os.path.join(base, "model", "evaluation_results.json")
    app.config["ERROR_ANALYSIS_PATH"] = os.path.join(base, "model", "error_analysis.json")

    # Load dataset and model
    app.config["DATAFRAME"] = pd.read_csv(app.config["DATA_PATH"])
    app.config["MODEL"] = CareerModel()

    # Precompute per-career lookups once at startup instead of re-scanning
    # the full DataFrame on every /predict or /skills request. The dataset
    # is small enough that this wasn't causing real slowness, but there's no
    # reason to redo an unchanging groupby on every request when it only
    # needs to happen once, right after the CSV loads.
    df = app.config["DATAFRAME"]
    skill_freq_by_career = {}
    interest_freq_by_career = {}
    salary_stats_by_career = {}
    for career, career_df in df.groupby("career_category"):
        all_skills = []
        for skills_str in career_df["skills"]:
            all_skills.extend(s.strip().lower() for s in skills_str.split(",") if s.strip())
        skill_freq_by_career[career] = pd.Series(all_skills).value_counts()

        all_interests = []
        for interests_str in career_df["interests"]:
            all_interests.extend(i.strip().lower() for i in interests_str.split(",") if i.strip())
        interest_freq_by_career[career] = pd.Series(all_interests).value_counts()

        salary_stats_by_career[career] = {
            "min_salary": int(career_df["salary_min"].min()),
            "max_salary": int(career_df["salary_max"].max()),
            "avg_min": int(career_df["salary_min"].mean()),
            "avg_max": int(career_df["salary_max"].mean()),
        }
    app.config["SKILL_FREQ_BY_CAREER"] = skill_freq_by_career
    app.config["INTEREST_FREQ_BY_CAREER"] = interest_freq_by_career
    app.config["SALARY_STATS_BY_CAREER"] = salary_stats_by_career

    # Load evaluation results if available
    app.config["EVAL_RESULTS"] = None
    if os.path.exists(app.config["EVAL_PATH"]):
        with open(app.config["EVAL_PATH"], "r") as f:
            app.config["EVAL_RESULTS"] = json.load(f)

    # Load error analysis (misclassified examples / low-confidence
    # predictions on the held-out test set) if available -- generated
    # separately by `python -m model.error_analysis` since it re-derives
    # the same train/test split rather than being computed on every app
    # startup.
    app.config["ERROR_ANALYSIS"] = None
    if os.path.exists(app.config["ERROR_ANALYSIS_PATH"]):
        with open(app.config["ERROR_ANALYSIS_PATH"], "r") as f:
            app.config["ERROR_ANALYSIS"] = json.load(f)

    # Shared state (in-memory) for demo purposes
    app.config["RECENT_PREDICTIONS"] = []

    return app


app = create_app()


@app.context_processor
def inject_globals():
    """Inject common template variables."""
    return {
        "user_name": "Learner",
        "current_year": datetime.utcnow().year,
    }


def _to_indian_number_format(n: int) -> str:
    """
    Format an integer using the Indian numbering system (lakh/crore grouping),
    e.g. 1234567 -> "12,34,567" instead of the Western "1,234,567".
    """
    s = str(abs(int(n)))
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
    return f"{sign}{grouped}"


@app.template_filter('currency')
def format_currency(value):
    """Format value as Indian Rupees with lakh/crore grouping, plus the LPA
    (Lakhs Per Annum) shorthand Indian job postings commonly use,
    e.g. 850000 -> "₹8,50,000 (8.5 LPA)"."""
    try:
        value = int(value)
        rupees = f"₹{_to_indian_number_format(value)}"
        lpa = value / 100000
        return f"{rupees} ({lpa:.1f} LPA)"
    except Exception:
        return value


@app.template_filter('lpa')
def format_lpa(value):
    """Format a rupee value in Lakhs Per Annum only (e.g. 850000 -> '8.5 LPA'),
    for places that just need the short form without the full rupee figure."""
    try:
        lakhs = float(value) / 100000
        return f"{lakhs:.1f} LPA"
    except Exception:
        return value


@app.template_filter('percentage')
def format_percentage(value):
    """Format value as percentage."""
    try:
        return f"{float(value)*100:.1f}%"
    except Exception:
        return value


# Note: the /predict and /skills forms now submit years of experience
# directly (0-10) instead of an Entry/Mid/Senior bucket, matching how the
# model was actually trained (experience is a numeric feature via
# StandardScaler in the ColumnTransformer -- see model/train_model.py). This
# removed the need for a bucket-to-years mapping entirely.


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def validate_input(skills: str, interests: str, education: str, experience: str) -> tuple:
    """
    Validate user input for career prediction.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    skills = skills.strip()
    interests = interests.strip()
    education = education.strip()
    experience = experience.strip()
    
    if not skills or len(skills) < 2:
        return False, "Please enter at least one skill."
    
    if not interests or len(interests) < 2:
        return False, "Please enter at least one interest."
    
    if not education:
        return False, "Please select an education level."
    
    if not experience:
        return False, "Please select your years of experience."

    try:
        experience_years = float(experience)
        if experience_years < 0 or experience_years > 60:
            return False, "Years of experience must be between 0 and 60."
    except ValueError:
        return False, "Years of experience must be a number."
    
    if len(skills) > 500:
        return False, "Skills input is too long. Please keep it under 500 characters."
    
    if len(interests) > 500:
        return False, "Interests input is too long. Please keep it under 500 characters."
    
    return True, ""


def get_matching_skills_for_career(career: str, user_skills: str) -> dict:
    """
    ===== DATASET-BASED LOOKUP (NOT ML PREDICTION) =====
    
    Matches user skills against dataset records for the career, using the
    per-career skill-frequency table precomputed once at startup (see
    create_app()) instead of re-scanning the DataFrame on every request.
    
    Args:
        career: Career category name
        user_skills: Comma-separated string of user skills
        
    Returns:
        dict with 'matching' and 'missing' skills
    """
    user_skills_set = {s.strip().lower() for s in user_skills.split(",") if s.strip()}

    skill_freq = app.config["SKILL_FREQ_BY_CAREER"].get(career)
    if skill_freq is None or skill_freq.empty:
        return {"matching": [], "missing": []}

    matching = [skill for skill in user_skills_set if skill in skill_freq.index]

    # Get most common skills for this career (from dataset), ranked by frequency
    top_skills = skill_freq.head(10).index.tolist()
    missing = [s for s in top_skills if s not in user_skills_set]
    
    return {
        "matching": sorted(matching)[:10],
        "missing": missing[:5],
    }


def get_matching_interests_for_career(career: str, user_interests: str) -> dict:
    """
    ===== DATASET-BASED LOOKUP (NOT ML PREDICTION) =====

    Same idea as get_matching_skills_for_career(), but for the `interests`
    field: shows which of the user's stated interests match interests
    commonly associated with the predicted career, using the per-career
    interest-frequency table precomputed once at startup.

    Args:
        career: Career category name
        user_interests: Comma-separated string of user interests

    Returns:
        dict with 'matching' interests (a simple, explainable lookup, not
        an ML prediction)
    """
    user_interests_set = {i.strip().lower() for i in user_interests.split(",") if i.strip()}

    interest_freq = app.config["INTEREST_FREQ_BY_CAREER"].get(career)
    if interest_freq is None or interest_freq.empty:
        return {"matching": []}

    matching = [interest for interest in user_interests_set if interest in interest_freq.index]

    return {"matching": sorted(matching)[:10]}


def get_salary_statistics(career: str) -> dict:
    """
    ===== DATASET-BASED LOOKUP (NOT ML PREDICTION) =====
    
    Get salary statistics from the dataset for a career, from the per-career
    table precomputed once at startup (see create_app()) instead of
    re-filtering the DataFrame on every request.
    
    Args:
        career: Career category name
        
    Returns:
        dict with salary statistics
    """
    return app.config["SALARY_STATS_BY_CAREER"].get(
        career, {"min_salary": 0, "max_salary": 0, "avg_min": 0, "avg_max": 0}
    )


# ============================================================================
# ROUTES
# ============================================================================


@app.route("/")
def root():
    """Home page."""
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    """Dashboard with dataset statistics."""
    df = app.config["DATAFRAME"]

    stats = {
        "total": len(df),
        "unique_careers": df["career_category"].nunique(),
        "min_salary": df["salary_min"].min(),
        "max_salary": df["salary_max"].max(),
    }

    recent = app.config.get("RECENT_PREDICTIONS", [])

    return render_template("dashboard.html", stats=stats, recent=recent)


@app.route("/evaluate")
def evaluate():
    """
    ===== MODEL EVALUATION PAGE =====
    
    Display comprehensive model evaluation metrics and comparison.
    This section shows the ML model's performance on the test set.
    """
    eval_results = app.config.get("EVAL_RESULTS", None)
    error_analysis = app.config.get("ERROR_ANALYSIS", None)

    if not eval_results:
        return render_template(
            "evaluate.html",
            eval_results=None,
            error_analysis=error_analysis,
            error="Model evaluation results not available. Please train the model first.",
        )

    return render_template("evaluate.html", eval_results=eval_results, error_analysis=error_analysis)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    """
    ===== CAREER PREDICTION ROUTE =====
    
    Components:
    - ML PREDICTION: best-performing classifier from training (see model/evaluation_results.json)
      via predict_proba() for genuine probabilities
    - DATASET LOOKUP: Gets matching skills and salary ranges from training data
    - RULE-BASED: Career roadmap is manually defined in model/roadmaps.py
    """
    form = {
        "skills": "",
        "interests": "",
        "education": "",
        "experience": "",
        "what_if_skills": ""
    }
    prediction = None
    what_if = None
    error = None

    if request.method == "POST":
        form["skills"] = request.form.get("skills", "").strip()
        form["interests"] = request.form.get("interests", "").strip()
        form["education"] = request.form.get("education", "").strip()
        form["experience"] = request.form.get("experience", "").strip()

        # Validate input
        is_valid, error_msg = validate_input(
            form["skills"], form["interests"], form["education"], form["experience"]
        )
        
        if not is_valid:
            error = error_msg
        else:
            try:
                # Experience is submitted directly as years (validated as numeric above)
                experience_num = float(form["experience"])

                # Handle what-if skills input
                raw_what_if = request.form.getlist("what_if_skills") or request.form.get(
                    "what_if_skills", ""
                )
                if isinstance(raw_what_if, list):
                    what_if_skills = [s.strip() for s in raw_what_if if s.strip()]
                    form["what_if_skills"] = ", ".join(what_if_skills)
                else:
                    what_if_skills = [s.strip() for s in raw_what_if.split(",") if s.strip()]
                    form["what_if_skills"] = raw_what_if

                model: CareerModel = app.config["MODEL"]

                # ===== ML PREDICTION (best-performing classifier from training) =====
                ranked = model.predict(
                    skills=form["skills"],
                    interests=form["interests"],
                    education=form["education"],
                    experience=experience_num,
                )

                main = ranked[0][0]
                main_score = ranked[0][1]
                alternatives = [
                    {"career": c, "score": s}
                    for c, s in ranked[1:4]
                    if s > 0
                ]

                # ===== DATASET-BASED LOOKUP (Skills, Interests, and Salary) =====
                skills_info = get_matching_skills_for_career(main, form["skills"])
                interests_info = get_matching_interests_for_career(main, form["interests"])
                salary_stats = get_salary_statistics(main)

                prediction = {
                    "main": main,
                    "score": main_score,
                    "alternatives": alternatives,
                    # Skills/interests are from dataset lookups, not the ML model
                    "matching_skills": skills_info["matching"],
                    "missing_skills": skills_info["missing"],
                    "matching_interests": interests_info["matching"],
                    "salary_min": salary_stats["min_salary"],
                    "salary_max": salary_stats["max_salary"],
                    # Rule-based roadmap
                    "roadmap": get_roadmap(main),
                }

                # ===== WHAT-IF ANALYSIS =====
                if form["what_if_skills"]:
                    what_if_skills_list = [
                        s.strip() for s in form["what_if_skills"].split(",") if s.strip()
                    ]
                    base_skills = [s.strip() for s in form["skills"].split(",") if s.strip()]
                    modified_skills = base_skills + [
                        s for s in what_if_skills_list if s not in base_skills
                    ]
                    modified_skills_str = ", ".join(modified_skills)

                    # ===== ML PREDICTION (modified input) =====
                    ranked_alt = model.predict(
                        skills=modified_skills_str,
                        interests=form["interests"],
                        education=form["education"],
                        experience=experience_num,
                    )

                    alt_main = ranked_alt[0][0]
                    alt_score = ranked_alt[0][1]

                    # ===== DATASET-BASED LOOKUP (alternative career) =====
                    alt_salary_stats = get_salary_statistics(alt_main)

                    what_if = {
                        "skills_added": what_if_skills_list,
                        "career_before": main,
                        "career_after": alt_main,
                        "score_before": main_score,
                        "score_after": alt_score,
                        "salary_before_min": salary_stats["min_salary"],
                        "salary_before_max": salary_stats["max_salary"],
                        "salary_after_min": alt_salary_stats["min_salary"],
                        "salary_after_max": alt_salary_stats["max_salary"],
                        "career_changed": alt_main != main,
                        "roadmap_after": get_roadmap(alt_main),
                    }

                # Update recent predictions
                recent = app.config.get("RECENT_PREDICTIONS", [])
                recent.insert(
                    0,
                    {
                        "career": main,
                        "score": main_score,
                        "skills": form["skills"],
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                )
                app.config["RECENT_PREDICTIONS"] = recent[:5]

            except Exception as e:
                error = f"An error occurred during prediction: {str(e)}"

    return render_template(
        "predict.html", form=form, prediction=prediction, what_if=what_if, error=error
    )


@app.route("/skills", methods=["GET", "POST"])
def skills():
    """
    ===== SKILL ANALYSIS ROUTE =====
    
    Components:
    - ML PREDICTION: best-performing classifier from training for career prediction
    - DATASET LOOKUP: Gets skills from training data for the predicted career
    """
    form = {"skills": "", "interests": "", "education": "", "experience": ""}
    recommendation = None
    error = None

    if request.method == "POST":
        form["skills"] = request.form.get("skills", "").strip()
        form["interests"] = request.form.get("interests", "").strip()
        form["education"] = request.form.get("education", "").strip()
        form["experience"] = request.form.get("experience", "").strip()

        # Validate input
        is_valid, error_msg = validate_input(
            form["skills"], form["interests"], form["education"], form["experience"]
        )
        
        if not is_valid:
            error = error_msg
        else:
            try:
                # Experience is submitted directly as years (validated as numeric above)
                experience_num = float(form["experience"])

                model: CareerModel = app.config["MODEL"]

                # ===== ML PREDICTION =====
                ranked = model.predict(
                    skills=form["skills"],
                    interests=form["interests"],
                    education=form["education"],
                    experience=experience_num,
                )

                career = ranked[0][0]
                score = ranked[0][1]

                # ===== DATASET-BASED LOOKUP =====
                skills_info = get_matching_skills_for_career(career, form["skills"])
                interests_info = get_matching_interests_for_career(career, form["interests"])
                salary_stats = get_salary_statistics(career)

                recommendation = {
                    "career": career,
                    "score": score,
                    "matching_skills": skills_info["matching"],
                    "missing_skills": skills_info["missing"],
                    "matching_interests": interests_info["matching"],
                    "salary_min": salary_stats["min_salary"],
                    "salary_max": salary_stats["max_salary"],
                }

            except Exception as e:
                error = f"An error occurred: {str(e)}"

    return render_template(
        "skills.html", form=form, recommendation=recommendation, error=error
    )


@app.route("/timeline", methods=["GET", "POST"])
def timeline():
    """
    ===== CAREER ROADMAP ROUTE =====
    
    This is RULE-BASED ONLY (not ML, not dataset-based).
    Roadmaps are manually defined in model/roadmaps.py
    """
    form = {"career": ""}
    roadmap = None
    careers = sorted(list(CAREER_ROADMAPS.keys()))

    if request.method == "POST":
        form["career"] = request.form.get("career", "").strip()
        if form["career"]:
            roadmap = get_roadmap(form["career"])

    return render_template("timeline.html", form=form, careers=careers, roadmap=roadmap)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
