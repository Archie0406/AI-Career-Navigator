# Career roadmaps for a 5-year growth plan.
# Extend this with more careers as needed.

CAREER_ROADMAPS = {
    "Data Analyst": [
        {"title": "Year 1", "description": "Build a foundation in SQL, Excel, and basic visualization tools."},
        {"title": "Year 2", "description": "Complete 2-3 data projects and publish a portfolio (Power BI/Tableau)."},
        {"title": "Year 3", "description": "Learn advanced analytics and basic machine learning techniques."},
        {"title": "Year 4", "description": "Focus on domain specialization (e.g., finance, healthcare)."},
        {"title": "Year 5", "description": "Mentor junior analysts and lead a cross-team analytics initiative."},
    ],
    "Machine Learning Engineer": [
        {"title": "Year 1", "description": "Learn Python, basic statistics, and machine learning fundamentals."},
        {"title": "Year 2", "description": "Build end-to-end models and learn deployment basics (Docker)."},
        {"title": "Year 3", "description": "Work with model monitoring and MLOps best practices."},
        {"title": "Year 4", "description": "Drive model optimization and production artifacts for scalability."},
        {"title": "Year 5", "description": "Lead ML initiatives and mentor other engineers."},
    ],
    "Software Engineer": [
        {"title": "Year 1", "description": "Master a primary language (Python/JavaScript) and version control."},
        {"title": "Year 2", "description": "Ship projects, build tests, and learn software architecture."},
        {"title": "Year 3", "description": "Master performance optimization and system design."},
        {"title": "Year 4", "description": "Lead or architect larger systems and mentor others."},
        {"title": "Year 5", "description": "Move toward senior/lead roles, owning cross-team components."},
    ],
    "Product Manager": [
        {"title": "Year 1", "description": "Learn product frameworks and conduct user research."},
        {"title": "Year 2", "description": "Run small experiments and establish metrics/KPIs."},
        {"title": "Year 3", "description": "Build cross-functional alignment and roadmaps."},
        {"title": "Year 4", "description": "Own a product area and deliver measurable impact."},
        {"title": "Year 5", "description": "Mentor PMs and shape product strategy at scale."},
    ],
    "Business Analyst": [
        {"title": "Year 1", "description": "Build strong Excel/SQL skills and create dashboards."},
        {"title": "Year 2", "description": "Work directly with stakeholders to understand business needs."},
        {"title": "Year 3", "description": "Learn automation and analytics best practices."},
        {"title": "Year 4", "description": "Lead analytics projects and improve decision processes."},
        {"title": "Year 5", "description": "Drive analytics strategy and mentor junior analysts."},
    ],
    "Back-End Developer": [
        {"title": "Year 1", "description": "Learn a server-side language (Python/Java/Node), databases, and REST APIs."},
        {"title": "Year 2", "description": "Build production services, write tests, and learn caching/queuing basics."},
        {"title": "Year 3", "description": "Master system design, scalability, and API security."},
        {"title": "Year 4", "description": "Own critical backend services and mentor teammates on architecture."},
        {"title": "Year 5", "description": "Lead backend platform strategy across teams."},
    ],
    "Front-End Developer": [
        {"title": "Year 1", "description": "Master HTML/CSS/JavaScript and a modern framework (React/Vue)."},
        {"title": "Year 2", "description": "Build accessible, responsive UIs and learn state management."},
        {"title": "Year 3", "description": "Focus on performance optimization and component architecture."},
        {"title": "Year 4", "description": "Lead front-end architecture decisions and design system work."},
        {"title": "Year 5", "description": "Mentor other engineers and shape front-end best practices org-wide."},
    ],
    "DevOps Engineer": [
        {"title": "Year 1", "description": "Learn Linux fundamentals, scripting, CI/CD basics, and Git workflows."},
        {"title": "Year 2", "description": "Automate deployments and adopt containerization (Docker/Kubernetes)."},
        {"title": "Year 3", "description": "Build infrastructure as code and improve observability/monitoring."},
        {"title": "Year 4", "description": "Own reliability engineering and incident response for larger systems."},
        {"title": "Year 5", "description": "Lead platform/DevOps strategy and mentor other engineers."},
    ],
    "UX Designer": [
        {"title": "Year 1", "description": "Learn user research methods, wireframing, and core design tools."},
        {"title": "Year 2", "description": "Ship end-to-end design projects and build a portfolio."},
        {"title": "Year 3", "description": "Develop expertise in usability testing and design systems."},
        {"title": "Year 4", "description": "Lead design for a product area and collaborate closely with PMs/engineers."},
        {"title": "Year 5", "description": "Mentor designers and shape design strategy at scale."},
    ],
}


def get_roadmap(career_name: str):
    return CAREER_ROADMAPS.get(career_name, [])
