# Data Science & AI Interview Preparation

A structured, self-paced study kit for data science, machine learning, GenAI, agentic systems, ML system design, and AI security interviews.

The repository combines long-form explanations, mathematical derivations, interview questions, active-recall flashcards, hands-on exercises, case studies, mock-interview drills, and templates for preparing project and behavioral answers.

## Start here

1. Begin with [`interview/00_INDEX_and_study_plan.md`](interview/00_INDEX_and_study_plan.md). It explains the recommended order, priorities, learning paths, and study schedule.
2. Read the theory guides in sequence, starting with [`interview/01_statistics_and_probability.md`](interview/01_statistics_and_probability.md).
3. Use [`interview/18_flashcards_active_recall.md`](interview/18_flashcards_active_recall.md) for active recall and spaced review.
4. Attempt the problems in [`interview/19_practice_problems_and_code.md`](interview/19_practice_problems_and_code.md) before viewing the solutions.
5. Practice applied reasoning with [`interview/20_case_studies_and_use_cases.md`](interview/20_case_studies_and_use_cases.md).
6. Rehearse answers with [`interview/21_mock_interview_and_progress_tracker.md`](interview/21_mock_interview_and_progress_tracker.md).
7. Personalize [`interview/22_project_deep_dives.md`](interview/22_project_deep_dives.md) and [`interview/23_behavioral_star_stories.md`](interview/23_behavioral_star_stories.md) with details you can accurately defend.

## Learning paths

| Goal | Suggested sequence |
| --- | --- |
| Data scientist | 01 → 02 → 03 → 05 → 06 → 07 → 20 |
| ML engineer | 03 → 04 → 06 → 08 → 09 → 13 → 19 |
| GenAI / LLM engineer | 03 → 10 → 11 → 12 → 13 → 14 → 19 |
| Agentic AI and security | 11 → 12 → 14 → 15 → 16 → 20 |
| Fast revision | 17 → 18 → selected problems in 19 → mock drills in 21 |

Use the [master index](interview/00_INDEX_and_study_plan.md) for the complete file-by-file map.

## What's included

### Core theory

- Statistics, probability, hypothesis testing, and A/B testing
- Machine-learning fundamentals, algorithms, ensembles, and boosting
- Model evaluation, feature engineering, explainability, and ranking
- Time-series forecasting
- SQL, PySpark, dbt, and data-engineering fundamentals
- MLOps, deployment, monitoring, and cloud concepts
- NLP, deep learning, transformers, and LLMs
- RAG, agents, agent evaluation, and ML system design
- Security fundamentals and AI-agent security

### Practice and revision

- Quick-fire review and certification talking points
- 270+ active-recall flashcards
- SQL, probability, statistics, derivation, and Python exercises
- Applied case studies and system-design prompts
- Mock-interview drills, rubrics, and progress tracking
- Project deep-dive and STAR-story templates

### Focused notebooks

The [`DS Questions/`](DS%20Questions/) directory contains topic-focused Jupyter notebooks, including a worked [linear-regression notebook](DS%20Questions/Linear%20Regression/linear-regression.ipynb).

## Repository structure

```text
interview/
  00_INDEX_and_study_plan.md       curriculum map and study plan
  01-07_*.md                      statistics, ML, evaluation, forecasting
  08-10_*.md                      data engineering, MLOps, NLP, deep learning
  11-16_*.md                      LLMs, RAG, agents, security, system design
  17-18_*.md                      revision and active-recall flashcards
  19-21_*.md                      practice, case studies, mock interviews
  22-23_*.md                      project and behavioral-answer templates
DS Questions/                      topic-focused Jupyter notebooks
pyproject.toml                     Python dependencies and project metadata
uv.lock                            locked dependency versions
```

## Run the notebooks

Reading the Markdown material requires no setup. To run the notebooks, use Python 3.12+ with [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync
uv run jupyter lab
```

Then open:

```text
DS Questions/Linear Regression/linear-regression.ipynb
```

The configured environment includes Jupyter support and common data-science libraries such as scikit-learn, XGBoost, LightGBM, Optuna, matplotlib, and seaborn.

## Practice the SQL exercises

The SQL exercises in `interview/19_practice_problems_and_code.md` use SQLite-flavored SQL and include their own sample schema and data. You can work through them with the SQLite CLI or Python's standard-library `sqlite3` module. No API keys or external services are required.

## Suggested study routine

1. Learn a topic from the theory guide.
2. Summarize it without looking at the notes.
3. Test yourself with the flashcards.
4. Solve a related problem or case study under a time limit.
5. Explain the solution aloud, including assumptions and trade-offs.
6. Record weak areas in the progress tracker and revisit them later.

## Important note

The project and behavioral sections are templates. Replace illustrative technologies, metrics, responsibilities, and stories with your own real experience before using them in an interview.
