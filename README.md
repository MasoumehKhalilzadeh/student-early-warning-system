# 🎓 Student Early Warning System

> A machine learning system that predicts whether a student is **At Risk** of failing or withdrawing from a course — enabling early intervention before it's too late.

[![Python](https://img.shields.io/badge/Python-3.9-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org)
[![GCP Cloud Run](https://img.shields.io/badge/GCP-Cloud%20Run-blue.svg)](https://cloud.google.com/run)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🌐 Live API

| | |
|---|---|
| **Base URL** | https://student-early-warning-115007201334.us-central1.run.app |
| **Interactive Docs** | https://student-early-warning-115007201334.us-central1.run.app/docs |
| **Health Check** | https://student-early-warning-115007201334.us-central1.run.app/health |

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Solution](#-solution)
- [Dataset](#-dataset)
- [Project Architecture](#-project-architecture)
- [Project Structure](#-project-structure)
- [Machine Learning Pipeline](#-machine-learning-pipeline)
- [Model Performance](#-model-performance)
- [Key Findings](#-key-findings)
- [API Documentation](#-api-documentation)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
- [Results Summary](#-results-summary)

---

## 🎯 Problem Statement

Universities lose thousands of students every year to withdrawal and failure. By the time a counselor notices a student is struggling, it is often too late to intervene effectively. The student has already disengaged, stopped submitting work, and mentally checked out.

**The question:** Can we identify at-risk students early enough to intervene — before they withdraw or fail?

---

## 💡 Solution

This project builds an end-to-end machine learning pipeline that:

1. Analyzes student behavior data from the first weeks of a course
2. Engineers meaningful features from raw interaction logs
3. Predicts whether a student is **At Risk** or **Not At Risk** with 88% accuracy
4. Serves predictions via a live REST API deployed on Google Cloud Platform

A counselor or instructor can send student data to the API at any point during a course and receive an instant risk assessment with a confidence score — enabling proactive, data-driven intervention.

---

## 📊 Dataset

**Open University Learning Analytics Dataset (OULAD)**

The OULAD dataset contains anonymized data from the UK's Open University — one of the world's largest distance learning institutions.

| Attribute | Value |
|---|---|
| Total Students | 32,593 |
| Number of Tables | 7 interconnected CSV files |
| Time Period | 2013–2014 academic years |
| License | CC BY 4.0 (free for research and commercial use) |
| Source | [analyse.kmi.open.ac.uk](https://analyse.kmi.open.ac.uk/open_dataset) |

### Dataset Tables

| Table | Description | Key Columns |
|---|---|---|
| `studentInfo.csv` | Demographics and final outcomes | gender, age_band, highest_education, final_result |
| `studentVle.csv` | Daily click activity on the learning platform | id_student, date, sum_click |
| `studentAssessment.csv` | Assessment scores and submission dates | score, date_submitted |
| `studentRegistration.csv` | Registration and withdrawal dates | date_registration, date_unregistration |
| `assessments.csv` | Assessment metadata | assessment_type, date, weight |
| `courses.csv` | Course information | module_presentation_length |
| `vle.csv` | Learning activity metadata | activity_type |

### Target Variable Distribution

```
Pass           12,361  (37.9%)
Withdrawn      10,156  (31.2%)
Fail            7,052  (21.6%)
Distinction     3,024  ( 9.3%)
```

**Binary classification (final model):**
```
At Risk     (Fail + Withdrawn)    17,208  (52.8%)
Not At Risk (Pass + Distinction)  15,385  (47.2%)
```

---

## 🏗 Project Architecture

```
Raw Data (7 CSV files)
        │
        ▼
┌───────────────────┐
│  Data Cleaning    │  Handle missing values, encode categories
│  & EDA            │  Visualize patterns, identify key signals
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Feature          │  Engineer 23 features from 7 raw tables
│  Engineering      │  Merge into single model-ready dataset
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Model Training   │  Train multiple models, handle class imbalance
│  & Evaluation     │  Select best model: Gradient Boosting (88%)
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  FastAPI          │  REST API with /predict endpoint
│  Application      │  Returns prediction + confidence score
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  GCP Cloud Run    │  Containerized with Docker
│  Deployment       │  Live public URL, auto-scaling
└───────────────────┘
```

---

## 📁 Project Structure

```
student-early-warning-system/
│
├── data/
│   ├── raw/                    ← OULAD CSV files (not tracked in git)
│   └── processed/              ← Engineered feature dataset (not tracked)
│
├── notebooks/
│   ├── 01_EDA.ipynb            ← Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb  ← Feature Engineering
│   └── 03_model_building.ipynb ← Model Training & Evaluation
│
├── api/
│   └── main.py                 ← FastAPI application
│
├── models/
│   ├── model_binary.pkl        ← Trained Gradient Boosting model
│   ├── scaler_binary.pkl       ← StandardScaler for feature scaling
│   └── features_binary.pkl     ← Feature column names
│
├── src/                        ← Utility scripts
├── Dockerfile                  ← Container configuration for GCP
├── .dockerignore               ← Files excluded from Docker build
├── requirements.txt            ← Python dependencies
└── README.md                   ← This file
```

---

## 🤖 Machine Learning Pipeline

### Phase 1: Exploratory Data Analysis

Explored all 7 tables to understand the data before modeling:

**Key findings from EDA:**
- VLE engagement showed a 10x difference between Distinction and Withdrawn students (median 2,200 vs 200 clicks)
- Students aged 0-35 had the highest withdrawal rates
- Lower educational background correlated with higher risk
- Gender showed minimal predictive power
- Missing unregistration dates (22,521 entries) indicated students who stayed enrolled — not a data quality issue

### Phase 2: Feature Engineering

Transformed 7 raw tables into a single model-ready dataset with 23 engineered features:

**Demographic Features:**
| Feature | Description |
|---|---|
| `gender_encoded` | Male=1, Female=0 |
| `disability_encoded` | Has disability=1, No=0 |
| `age_encoded` | Age group as ordered number (0-35=0, 35-55=1, 55+=2) |
| `education_encoded` | Education level as ordered number (0=None to 4=Postgraduate) |

**VLE Engagement Features:**
| Feature | Description |
|---|---|
| `total_clicks` | Total platform interactions across entire course |
| `avg_daily_clicks` | Mean clicks per active day |
| `active_days` | Number of days with any platform activity |
| `max_clicks_day` | Peak single-day engagement |
| `early_clicks` | Clicks in first 30 days (early warning signal) |
| `early_engagement_ratio` | Early clicks / total clicks |
| `click_consistency` | Total clicks / active days |

**Assessment Performance Features:**
| Feature | Description |
|---|---|
| `avg_score` | Mean score across all assessments |
| `max_score` | Best assessment performance |
| `min_score` | Worst assessment performance |
| `score_range` | max_score - min_score (consistency measure) |
| `num_assessments` | Total assessments completed |
| `num_late_submissions` | Count of late submissions |
| `late_submission_rate` | Proportion of late submissions |
| `assessment_completion_rate` | Assessments completed / total possible |

**Registration Features:**
| Feature | Description |
|---|---|
| `avg_registration_date` | How early student registered (negative = before course start) |
| `struggle_score` | Composite risk score combining late submissions, low scores, and low early engagement |

> **Data Leakage Note:** An initial feature `stayed_enrolled` was identified as causing data leakage — it directly encoded withdrawal status (the target variable). Removing it dropped accuracy from a misleading 77% to an honest 66%, which was then improved through better features and modeling to reach 88%.

### Phase 3: Model Training

**Train/Test Split:** 80% training (26,074 students) / 20% testing (6,519 students) with stratification to preserve class distribution.

**Models Evaluated:**

| Model | Accuracy | At Risk F1 | Notes |
|---|---|---|---|
| Random Forest | 66% | 0.69 | Good baseline, no data leakage |
| Gradient Boosting | 67% | 0.70 | Best 4-class performance |
| Logistic Regression | 60% | 0.68 | Most interpretable |
| **Binary Gradient Boosting** | **88%** | **0.89** | **Final model** |

**Class Imbalance Handling:**
Applied SMOTE (Synthetic Minority Oversampling Technique) for the 4-class model to balance Distinction class (2,419 examples) against Pass class (9,889 examples):

```
Before SMOTE: Distinction=2,419  Fail=5,641  Pass=9,889  Withdrawn=8,125
After SMOTE:  Distinction=9,889  Fail=9,889  Pass=9,889  Withdrawn=9,889
```

**Binary Classification Insight:**
Simplifying to binary (At Risk vs Not At Risk) produced near-perfect class balance (52.8% vs 47.2%) without needing SMOTE, and pushed accuracy from 67% to 88%. This approach is also more actionable for real-world intervention programs.

---

## 📈 Model Performance

### Final Model: Binary Gradient Boosting Classifier

```
Overall Accuracy:      88%
Random Chance:         50% (binary problem)
Improvement:           +38 percentage points over random

              precision    recall  f1-score   support
 Not At Risk       0.85      0.91      0.88      3077
     At Risk       0.92      0.86      0.89      3442
    accuracy                           0.88      6519
```

### Confusion Matrix Results

```
                  Predicted
                  Not At Risk   At Risk
Actual Not Risk      2,812        265
Actual At Risk         495      2,947
```

**In plain terms:**
- Correctly identified **2,947** at-risk students out of 3,442 (86% recall)
- Only missed **495** at-risk students (14% missed)
- Incorrectly flagged **265** safe students as at-risk (8% false alarm rate)

### Feature Importance

```
1. assessment_completion_rate    34.0%  ← strongest predictor
2. num_assessments               30.0%  ← engagement depth
3. early_engagement_ratio        11.0%  ← first 30 days behavior
4. avg_score                      8.0%  ← academic performance
5. min_score                      3.0%  ← worst performance
6. struggle_score                 2.5%
7. total_clicks                   2.3%
8. active_days                    1.8%
   ... (demographic features had minimal importance)
```

---

## 🔍 Key Findings

1. **Assessment completion is the strongest predictor (64% combined importance)** — Whether a student completes assessments matters far more than demographics like age, gender, or disability. Showing up and submitting work predicts success better than who you are.

2. **Early engagement predicts final outcomes** — VLE activity in the first 30 days (early_engagement_ratio) was the 3rd most important feature, confirming the system can identify at-risk students very early in a course.

3. **Engagement matters more than performance** — A student who submits and scores 60% is far less at risk than one who doesn't submit at all. Participation predicts outcomes better than grades alone.

4. **Demographics have minimal predictive power** — Age, gender, and disability were among the least important features, suggesting intervention programs should focus on engagement behavior rather than demographic targeting.

5. **10x engagement gap between outcomes** — Distinction students had a median of 2,200 VLE clicks versus 200 for withdrawn students — a clear, measurable behavioral signal detectable weeks before final outcomes.

---

## 🚀 API Documentation

### Base URL
```
https://student-early-warning-115007201334.us-central1.run.app
```

### Endpoints

#### GET /
Returns API status.

```json
{
  "message": "Student Early Warning System API",
  "version": "1.0.0",
  "status": "running"
}
```

#### GET /health
Health check endpoint.

```json
{"status": "healthy"}
```

#### POST /predict
Returns at-risk prediction for a student.

**Request Body:**
```json
{
  "gender_encoded": 1.0,
  "disability_encoded": 0.0,
  "age_encoded": 0.0,
  "education_encoded": 2.0,
  "num_of_prev_attempts": 0.0,
  "studied_credits": 60.0,
  "total_clicks": 200.0,
  "avg_daily_clicks": 2.5,
  "active_days": 10.0,
  "max_clicks_day": 20.0,
  "early_clicks": 50.0,
  "avg_score": 35.0,
  "max_score": 50.0,
  "min_score": 10.0,
  "num_assessments": 2.0,
  "num_late_submissions": 3.0,
  "late_submission_rate": 0.8,
  "avg_registration_date": -10.0,
  "click_consistency": 2.0,
  "early_engagement_ratio": 0.1,
  "assessment_completion_rate": 0.2,
  "score_range": 40.0,
  "struggle_score": 0.8
}
```

**Response:**
```json
{
  "prediction": "At Risk",
  "confidence": "98.20%",
  "at_risk_probability": "98.20%",
  "recommendation": "Immediate intervention recommended"
}
```

### Example: High Performing Student

```json
{
  "prediction": "Not At Risk",
  "confidence": "91.50%",
  "at_risk_probability": "8.50%",
  "recommendation": "Student appears to be on track"
}
```

---

## 🛠 Tech Stack

| Category | Technology | Purpose |
|---|---|---|
| Language | Python 3.9 | Core development language |
| Data Processing | pandas, numpy | Data manipulation and analysis |
| Visualization | matplotlib, seaborn | EDA charts and feature importance plots |
| Machine Learning | scikit-learn | Model training, scaling, evaluation |
| Class Balancing | imbalanced-learn | SMOTE oversampling |
| API Framework | FastAPI | REST API with automatic documentation |
| API Server | uvicorn | ASGI server for FastAPI |
| Containerization | Docker | Application packaging for deployment |
| Cloud Platform | GCP Cloud Run | Serverless container deployment |
| Version Control | Git / GitHub | Code versioning and portfolio |
| IDE | VS Code | Development environment |

---

## 🚦 Getting Started

### Prerequisites
- Python 3.9+
- Git
- VS Code (recommended)

### Local Setup

```bash
# Clone the repository
git clone https://github.com/YOURUSERNAME/student-early-warning-system.git
cd student-early-warning-system

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Download Dataset

1. Go to [analyse.kmi.open.ac.uk/open_dataset](https://analyse.kmi.open.ac.uk/open_dataset) or [Kaggle OULAD](https://www.kaggle.com/datasets/anlgrbz/student-demographics-online-education-dataoulad)
2. Download all CSV files
3. Place them in `data/raw/`

### Run the Notebooks

```bash
# Start Jupyter
jupyter notebook

# Run in order:
# 1. notebooks/01_EDA.ipynb
# 2. notebooks/02_feature_engineering.ipynb
# 3. notebooks/03_model_building.ipynb
```

### Run the API Locally

```bash
cd api
uvicorn main:app --reload --port 8000
```

Visit `http://localhost:8000/docs` for interactive documentation.

### Every Time You Return

```bash
# 1. Open VS Code → Open Folder → student-early-warning-system
# 2. Open terminal (Control + `)
source venv/bin/activate
git pull
# 3. Select Python (venv) kernel in notebooks
```

---

## 📊 Results Summary

| Metric | Value |
|---|---|
| Dataset Size | 32,593 students |
| Features Engineered | 23 features from 7 raw tables |
| Best Model | Gradient Boosting Classifier (binary) |
| Overall Accuracy | 88% |
| At Risk Precision | 92% |
| At Risk Recall | 86% |
| Improvement Over Random | +38 percentage points |
| Deployment | GCP Cloud Run (live 24/7) |

**Real-world impact:** In a cohort of 1,000 students, this system would correctly identify 454 of 528 at-risk students before they withdraw or fail — giving counselors actionable intelligence weeks before traditional methods would detect the problem.

---

## 👤 Author

**Masoumeh Khalilzadeh**
- MS Statistics | MS Mathematics (Statistics concentration) | BS Statistics
- Data Analyst at SJSU

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- Dataset: Kuzilek J., Hlosta M., Zdrahal Z. [Open University Learning Analytics Dataset](https://www.nature.com/articles/sdata2017171) Sci. Data 4:170171 doi: 10.1038/sdata.2017.171 (2017)
- Deployed on Google Cloud Platform Cloud Run
