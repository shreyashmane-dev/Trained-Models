# AI Governance Platform Demo Assets (with Monthly Drift Batches)

This directory contains four complete machine learning demo projects engineered for demonstrating AI Governance Platform capabilities. The assets are designed for testing **model evaluation, SHAP explainability, governance analysis, dataset/feature drift detection, model stability trendlines, and audit logging**.

To ensure 100% plug-and-play compatibility with standard REST APIs (e.g. FastAPI backends) and SHAP explainability out-of-the-box, all categorical features have been encoded as numeric representations (integers). This prevents serialization and runtime conversion exceptions (`ValueError: could not convert string to float`) in `scikit-learn` and `shap` libraries.

---

## Directory Structure

Each project folder includes 5 sequential datasets representing different time periods (`month_1.csv` to `month_5.csv`). This allows you to construct charts showing **model performance decay, feature distribution shifts (PSI), and concept drift trendlines** over time.

```text
demo-assets/
│
├── requirements.txt            # Python dependencies (pandas, scikit-learn, joblib, shap)
├── README.md                   # Platform asset documentation (this file)
│
├── loan-approval/              # Case 1: Fairness & Stability
│   ├── dataset.csv             # Full training & testing data (1500 rows)
│   ├── month_1.csv             # Month 1 production batch (1200 rows) (Baseline Reference)
│   ├── month_2.csv             # Month 2 production batch (1200 rows)
│   ├── month_3.csv             # Month 3 production batch (1200 rows) (Income Drift)
│   ├── month_4.csv             # Month 4 production batch (1200 rows) (Credit History Drift)
│   ├── month_5.csv             # Month 5 production batch (1200 rows) (Concept Policy Drift)
│   ├── baseline.csv            # Reference deployment dataset (identical to month_1.csv)
│   ├── current.csv             # Active production dataset (identical to month_5.csv)
│   ├── model.pkl               # Serialized RandomForestClassifier
│   └── train_model.py          # Data generation and model training script
│
├── heart-disease/              # Case 2: Explainability & Medical Transparency
│   ├── dataset.csv             # 1500 rows
│   ├── month_1.csv ... month_5.csv # 5 sequential monthly batches (1200 rows each)
│   ├── baseline.csv            # Identical to month_1.csv
│   ├── current.csv             # Identical to month_5.csv
│   ├── model.pkl               # Serialized RandomForestClassifier
│   └── train_model.py          # Data generation and model training script
│
├── employee-attrition/         # Case 3: Dataset Drift Detection (Salary Shift)
│   ├── dataset.csv             # 1500 rows
│   ├── month_1.csv ... month_5.csv # 5 sequential monthly batches (1200 rows each)
│   ├── baseline.csv            # Identical to month_1.csv
│   ├── current.csv             # Identical to month_5.csv
│   ├── model.pkl               # Serialized RandomForestClassifier
│   └── train_model.py          # Data generation and model training script
│
└── insurance-fraud/            # Case 4: Risk Scoring & Fraud Explainability
    ├── dataset.csv             # 1500 rows
    ├── month_1.csv ... month_5.csv # 5 sequential monthly batches (1200 rows each)
    ├── baseline.csv            # Identical to month_1.csv
    ├── current.csv             # Identical to month_5.csv
    ├── model.pkl               # Serialized RandomForestClassifier
    └── train_model.py          # Data generation and model training script
```

---

## Drift & Stability Scenario Documentation

Running covariate drift tools (like Population Stability Index, Kolmogorov-Smirnov, or Wasserstein Distance) and plotting performance metrics over the monthly batches will show the following simulated trends:

### 1. Loan Approval Prediction

* **Target Variable**: `Loan_Status` (`0` = Rejected, `1` = Approved)
* **Governance/Stability Scenario**:
  - **Months 1 & 2 (Stable)**: Distributions and metrics match baseline model expectations.
  - **Month 3 (Feature Drift)**: `ApplicantIncome` distribution shifts higher (simulating high-inflation adjustments in the market).
  - **Month 4 (Covariate Drift)**: `CreditHistory` rates decline (proportion of good history drop from 84% to 55%, simulating recession impact).
  - **Month 5 (Concept Drift & Performance Decay)**: The underlying lending criteria shifts (CreditHistory becomes less predictive, while Education becomes more heavily weighted). Model accuracy drops noticeably from **~0.78** down to **~0.69**.

---

### 2. Heart Disease Prediction

* **Target Variable**: `HeartDisease` (`0` = No disease, `1` = Disease present)
* **Governance/Stability Scenario**:
  - **Months 1 & 2 (Stable)**: Standard patient profile entries.
  - **Month 3 (Demographic Drift)**: Patient cohort shifts older (`Age` mean shifts from 54 to 62 years, reflecting seasonal geriatric clinic referrals).
  - **Month 4 (Risk Feature Drift)**: Fasting blood sugar (`FastingBS`) increases from 15% probability to 35%.
  - **Month 5 (Concept Drift)**: Minor diagnostic parameter weighting shift based on updated clinical measurement tools.

---

### 3. Employee Attrition Prediction

* **Target Variable**: `Attrition` (`0` = Stayed, `1` = Left)
* **Governance/Stability Scenario**:
  - **Month 1 (Stable)**: Initial organizational baseline.
  - **Month 2 (Overtime Shift)**: Overtime rates begin rising due to seasonal product releases (from 30% to 45% probability).
  - **Month 3 (Gradual Salary Drop)**: Shift starts as the firm prioritizes entry-level recruitment.
  - **Month 4 (Major Salary Drift)**: Major corporate restructuring shifts mean salaries from $85k to $62k (strongly flags covariate drift).
  - **Month 5 (Burnout/Concept Drift)**: Attrition rate spikes as a combined result of lower salaries and sustained overtime. Model accuracy decays from **~0.75** down to **~0.65**.

---

### 4. Insurance Fraud Detection

* **Target Variable**: `Fraudulent` (`0` = Legitimate, `1` = Fraudulent)
* **Governance/Stability Scenario**:
  - **Months 1 & 2 (Stable)**: Standard claims intake.
  - **Month 3 (Portfolio Drift)**: Insured vehicle portfolio shifts older (`VehicleAge` mean shifts from 7 to 11 years).
  - **Month 4 (Extreme Event Drift)**: Catastrophic weather event causes a major spike in claim amounts (`ClaimAmount` mean shifts significantly higher).
  - **Month 5 (Concept Drift)**: Shift in fraudulent behavior profile (fraud ring begins targeting minor severity claims with higher frequency).

---

## Setup & Regeneration

To install the required dependencies and regenerate all monthly datasets, run:

```bash
# Install dependencies
pip install -r requirements.txt

# Run training scripts
python loan-approval/train_model.py
python heart-disease/train_model.py
python employee-attrition/train_model.py
python insurance-fraud/train_model.py
```
Each training script will regenerate the 5 monthly datasets, baseline/current files, train a fresh scikit-learn `RandomForestClassifier` (preserving feature names), and output `model.pkl`.
