"""
Synthetic dataset generator for the Career Path Navigator project.

Builds each row from 3 skill/interest pools per career -- PRIMARY
(career-typical), OVERLAP (shared with 1-2 adjacent careers, e.g. Data
Analyst <-> Business Analyst both use sql/excel), and UNIVERSAL (generic,
e.g. communication) -- plus random cross-career noise skills. This keeps
any single skill from deterministically implying one career, and class
sizes are deliberately imbalanced (see CAREER_TARGET_COUNTS). No career
name is ever written into skills/interests text, so there's no target
leakage. See METHODOLOGY.md item 11 for why this design was needed.

Run: python -m model.generate_dataset  (rewrites data/jobs_sample.csv)
"""

import pathlib
import random

import numpy as np
import pandas as pd

RANDOM_STATE = 42

_MODEL_DIR = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _MODEL_DIR.parent
_DEFAULT_OUT = _PROJECT_ROOT / "data" / "jobs_sample.csv"

# Deliberately imbalanced class distribution -- not sourced from real
# labor-market data, just a designed synthetic spread for practice.
CAREER_TARGET_COUNTS = {
    "Software Engineer": 300,
    "Data Analyst": 260,
    "Front-End Developer": 240,
    "Back-End Developer": 220,
    "Machine Learning Engineer": 200,
    "Business Analyst": 180,
    "DevOps Engineer": 160,
    "Product Manager": 140,
    "UX Designer": 120,
}
TOTAL_RECORDS = sum(CAREER_TARGET_COUNTS.values())  # 1820

CAREERS = list(CAREER_TARGET_COUNTS.keys())

# ---------------------------------------------------------------------------
# 2. SKILL POOLS
# ---------------------------------------------------------------------------

PRIMARY_SKILLS = {
    "Data Analyst": [
        "dax", "power bi", "vlookup", "pivot tables", "google sheets", "looker",
        "qlik", "data storytelling", "regression analysis", "data warehousing",
        "etl", "a/b testing", "kpis", "market research", "charts",
    ],
    "Business Analyst": [
        "business process improvement", "gap analysis", "financial modeling",
        "business case development", "use case documentation",
        "workflow automation", "risk analysis", "process mapping",
        "requirements gathering", "stakeholder interviews",
    ],
    "Machine Learning Engineer": [
        "deep learning", "pytorch", "tensorflow", "keras", "computer vision",
        "nlp", "reinforcement learning", "mlops", "model deployment",
        "hyperparameter tuning", "xgboost", "scikit-learn", "linear algebra",
        "model evaluation",
    ],
    "Software Engineer": [
        "object-oriented design", "software architecture", "code review",
        "unit testing", "refactoring", "concurrency", "design patterns",
        "c++", "c#", "performance tuning",
    ],
    "Front-End Developer": [
        "react", "vue", "angular", "svelte", "css animations", "sass",
        "tailwind css", "redux", "next.js", "storybook", "ui components",
        "cross-browser testing", "typescript", "progressive web apps",
    ],
    "Back-End Developer": [
        "node.js", "spring boot", "django", "grpc", "message queues", "kafka",
        "database indexing", "sharding", "load balancing", "caching",
        "authentication", "mongodb", "postgresql", "redis",
    ],
    "DevOps Engineer": [
        "kubernetes", "terraform", "ansible", "helm", "prometheus", "grafana",
        "infrastructure as code", "site reliability", "incident response",
        "monitoring", "networking", "logging systems",
    ],
    "Product Manager": [
        "go-to-market strategy", "pricing strategy", "okrs", "backlog grooming",
        "competitive analysis", "product analytics", "product strategy",
        "prioritization", "metrics",
    ],
    "UX Designer": [
        "heuristic evaluation", "card sorting", "usability",
        "interaction design", "information architecture", "journey mapping",
        "visual design", "design critique", "sketch",
    ],
}

OVERLAP_SKILLS = {
    ("Data Analyst", "Business Analyst"): [
        "sql", "excel", "data visualization", "reporting",
        "stakeholder communication", "dashboards", "tableau",
    ],
    ("Data Analyst", "Machine Learning Engineer"): [
        "python", "statistics", "pandas", "numpy", "data cleaning",
        "machine learning",
    ],
    ("Machine Learning Engineer", "Software Engineer"): [
        "python", "algorithms", "data structures", "git", "debugging",
    ],
    ("Software Engineer", "Back-End Developer"): [
        "git", "testing", "api design", "data structures", "design patterns",
        "debugging",
    ],
    ("Back-End Developer", "DevOps Engineer"): [
        "docker", "cloud", "linux", "ci/cd", "api design", "databases",
    ],
    ("Front-End Developer", "UX Designer"): [
        "figma", "css", "prototyping", "user research", "design systems",
    ],
    ("Front-End Developer", "Software Engineer"): [
        "javascript", "git", "testing", "debugging", "html",
    ],
    ("Product Manager", "Business Analyst"): [
        "stakeholder management", "requirements gathering", "market analysis",
        "roadmap", "communication",
    ],
    ("Product Manager", "UX Designer"): [
        "user research", "personas", "customer discovery",
    ],
    ("DevOps Engineer", "Software Engineer"): [
        "git", "ci/cd", "scripting", "testing", "linux",
    ],
}

UNIVERSAL_SKILLS = [
    "communication", "problem solving", "teamwork", "agile", "documentation",
    "git", "presentation skills", "critical thinking",
]

# ---------------------------------------------------------------------------
# 3. INTEREST POOLS (same primary/overlap/universal structure, smaller)
# ---------------------------------------------------------------------------
PRIMARY_INTERESTS = {
    "Data Analyst": [
        "dashboards", "forecasting", "data storytelling",
        "business intelligence", "market research", "kpis",
    ],
    "Business Analyst": [
        "process improvement", "financial analysis", "risk management",
        "requirements analysis",
    ],
    "Machine Learning Engineer": [
        "artificial intelligence", "predictive modeling", "computer vision",
        "nlp research", "model optimization",
    ],
    "Software Engineer": [
        "system design", "clean code", "software craftsmanship",
    ],
    "Front-End Developer": [
        "ui development", "interactive web experiences", "responsive design",
        "web performance",
    ],
    "Back-End Developer": [
        "distributed systems", "api design", "scalability",
        "system reliability",
    ],
    "DevOps Engineer": [
        "automation", "infrastructure", "site reliability", "cloud computing",
    ],
    "Product Manager": [
        "product strategy", "market fit", "customer needs", "growth",
    ],
    "UX Designer": [
        "human-centered design", "usability", "visual storytelling",
        "accessibility",
    ],
}

OVERLAP_INTERESTS = {
    ("Data Analyst", "Business Analyst"): ["reporting", "analytics", "visualization"],
    ("Data Analyst", "Machine Learning Engineer"): ["statistics", "data analysis", "pattern recognition"],
    ("Machine Learning Engineer", "Software Engineer"): ["programming", "algorithms"],
    ("Software Engineer", "Back-End Developer"): ["backend systems", "engineering"],
    ("Back-End Developer", "DevOps Engineer"): ["cloud", "infrastructure"],
    ("Front-End Developer", "UX Designer"): ["design", "user experience"],
    ("Front-End Developer", "Software Engineer"): ["web development", "coding"],
    ("Product Manager", "Business Analyst"): ["strategy", "stakeholder collaboration"],
    ("Product Manager", "UX Designer"): ["user research", "customer feedback"],
    ("DevOps Engineer", "Software Engineer"): ["automation", "systems"],
}

UNIVERSAL_INTERESTS = [
    "technology", "innovation", "learning new tools", "teamwork",
    "problem solving",
]

EDUCATION_LEVELS = ["12th pass", "diploma", "bachelor's degree", "master's degree", "phd"]
EDUCATION_WEIGHTS = [0.05, 0.15, 0.50, 0.25, 0.05]

EXPERIENCE_BINS = [(0, 1), (1, 3), (3, 6), (6, 10), (10, 15)]
EXPERIENCE_BIN_WEIGHTS = [0.20, 0.30, 0.25, 0.15, 0.10]

BASE_SALARY = {
    "Data Analyst": 420_000,
    "Business Analyst": 480_000,
    "Machine Learning Engineer": 650_000,
    "Software Engineer": 550_000,
    "Front-End Developer": 450_000,
    "Back-End Developer": 500_000,
    "DevOps Engineer": 600_000,
    "Product Manager": 800_000,
    "UX Designer": 430_000,
}
SALARY_PER_YEAR_EXPERIENCE = 55_000


def _adjacent_overlap_pools(pool: dict, career: str):
    pools = []
    for (a, b), items in pool.items():
        if career in (a, b):
            pools.append(items)
    return pools


def _sample_unique(rng: random.Random, pool, k):
    k = min(k, len(pool))
    if k <= 0:
        return []
    return rng.sample(pool, k)


def _generate_skills_or_interests(
    rng: random.Random,
    career: str,
    primary_pool: dict,
    overlap_pool: dict,
    universal_pool: list,
    n_primary_range,
    n_overlap_range,
    n_universal_range,
):
    tokens = []
    tokens += _sample_unique(rng, primary_pool[career], rng.randint(*n_primary_range))

    adjacent_pools = _adjacent_overlap_pools(overlap_pool, career)
    if adjacent_pools:
        combined_overlap = sorted(set(item for pool in adjacent_pools for item in pool))
        tokens += _sample_unique(rng, combined_overlap, rng.randint(*n_overlap_range))

    tokens += _sample_unique(rng, universal_pool, rng.randint(*n_universal_range))

    seen = set()
    unique_tokens = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            unique_tokens.append(t)
    rng.shuffle(unique_tokens)
    return unique_tokens


def _sample_experience(rng: random.Random) -> int:
    idx = rng.choices(range(len(EXPERIENCE_BINS)), weights=EXPERIENCE_BIN_WEIGHTS, k=1)[0]
    lo, hi = EXPERIENCE_BINS[idx]
    return rng.randint(lo, hi)


def _sample_education(rng: random.Random) -> str:
    return rng.choices(EDUCATION_LEVELS, weights=EDUCATION_WEIGHTS, k=1)[0]


def _sample_salary(rng: random.Random, career: str, experience: int):
    base = BASE_SALARY[career] + experience * SALARY_PER_YEAR_EXPERIENCE
    noise = rng.uniform(0.85, 1.15)
    salary_min = base * noise
    ratio = rng.uniform(2.0, 2.4)
    salary_max = salary_min * ratio
    return round(salary_min, -3), round(salary_max, -3)


def _n_primary_count(rng: random.Random) -> int:
    # Some rows get zero career-specific skills (relying on overlap/universal
    # tokens only), so no single skill is a deterministic tell.
    return rng.choices([0, 1, 2], weights=[0.62, 0.30, 0.08], k=1)[0]


def _make_row(rng, career):
    n_primary_skills = _n_primary_count(rng)
    n_primary_interests = _n_primary_count(rng)
    skills = _generate_skills_or_interests(
        rng, career, PRIMARY_SKILLS, OVERLAP_SKILLS, UNIVERSAL_SKILLS,
        n_primary_range=(n_primary_skills, n_primary_skills),
        n_overlap_range=(2, 5), n_universal_range=(0, 2),
    )
    interests = _generate_skills_or_interests(
        rng, career, PRIMARY_INTERESTS, OVERLAP_INTERESTS, UNIVERSAL_INTERESTS,
        n_primary_range=(n_primary_interests, n_primary_interests),
        n_overlap_range=(1, 3), n_universal_range=(0, 1),
    )

    # Cross-career noise: a real candidate sometimes has one odd skill from
    # an unrelated career.
    if rng.random() < 0.65:
        other_career = rng.choice([c for c in CAREERS if c != career])
        noise_skill = _sample_unique(rng, PRIMARY_SKILLS[other_career], rng.choice([1, 2, 2]))
        skills = skills + [s for s in noise_skill if s not in skills]
    if rng.random() < 0.45:
        other_career = rng.choice([c for c in CAREERS if c != career])
        noise_interest = _sample_unique(rng, PRIMARY_INTERESTS[other_career], 1)
        interests = interests + [i for i in noise_interest if i not in interests]

    if not skills:
        skills = _sample_unique(rng, UNIVERSAL_SKILLS, 2)
    if not interests:
        interests = _sample_unique(rng, UNIVERSAL_INTERESTS, 1)

    education = _sample_education(rng)
    experience = _sample_experience(rng)
    salary_min, salary_max = _sample_salary(rng, career, experience)

    return {
        "skills": ", ".join(skills),
        "interests": ", ".join(interests),
        "education": education,
        "experience": experience,
        "career_category": career,
        "salary_min": int(salary_min),
        "salary_max": int(salary_max),
    }


def generate_dataset(seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = random.Random(seed)
    np.random.seed(seed)

    rows = []
    for career, target_count in CAREER_TARGET_COUNTS.items():
        for _ in range(target_count):
            rows.append(_make_row(rng, career))

    df = pd.DataFrame(rows)

    # Remove any exact duplicate rows that happened to be sampled (checked,
    # not assumed) and top up each affected career back to its target count,
    # rather than silently shrinking the dataset below its intended size.
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    if removed:
        print(f"Removed {removed} exact-duplicate rows generated by chance; topping up...")
        extra_rows = []
        for career in CAREERS:
            deficit = CAREER_TARGET_COUNTS[career] - int((df["career_category"] == career).sum())
            for _ in range(deficit):
                extra_rows.append(_make_row(rng, career))
        df = pd.concat([df, pd.DataFrame(extra_rows)], ignore_index=True)
        df = df.drop_duplicates().reset_index(drop=True)

    # Shuffle row order so the CSV isn't grouped by career.
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def main(out_path: str = None):
    out_path = pathlib.Path(out_path) if out_path else _DEFAULT_OUT
    df = generate_dataset()

    print("=" * 70)
    print("DATASET GENERATION SUMMARY")
    print("=" * 70)
    print(f"Total records: {len(df)}")
    print(f"Duplicate rows remaining: {df.duplicated().sum()}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print("\nClass distribution:")
    dist = df["career_category"].value_counts()
    for career, count in dist.items():
        print(f"  {career:28s} {count:4d}  ({count/len(df)*100:5.1f}%)")
    print(f"\nClass balance ratio (min/max): {dist.min()/dist.max():.3f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
