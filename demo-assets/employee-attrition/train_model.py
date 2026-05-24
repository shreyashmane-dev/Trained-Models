import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

def generate_attrition_data(num_rows, seed=42):
    """
    Generates training synthetic employee attrition dataset with baseline features.
    """
    np.random.seed(seed)
    
    age = np.random.normal(37, 9, size=num_rows).astype(int)
    age = np.clip(age, 18, 60)
    
    salary = np.random.normal(85000, 25000, size=num_rows).astype(int)
    salary = np.clip(salary, 25000, 200000)
    
    department = np.random.choice([0, 1, 2], size=num_rows, p=[0.30, 0.60, 0.10])
    job_satisfaction = np.random.choice([1, 2, 3, 4], size=num_rows, p=[0.15, 0.25, 0.35, 0.25])
    overtime = np.random.choice([0, 1], size=num_rows, p=[0.70, 0.30])
    
    max_years = age - 18
    years_at_company = np.zeros(num_rows)
    for i in range(num_rows):
        if max_years[i] <= 1:
            years_at_company[i] = 1
        else:
            years_at_company[i] = np.random.choice(range(1, min(int(max_years[i]), 21)))
    years_at_company = years_at_company.astype(int)
    
    work_life_balance = np.random.choice([1, 2, 3, 4], size=num_rows, p=[0.10, 0.25, 0.45, 0.20])
    distance_from_home = np.random.exponential(scale=8.0, size=num_rows).astype(int)
    distance_from_home = np.clip(distance_from_home, 1, 30)
    
    df = pd.DataFrame({
        'Age': age,
        'Salary': salary,
        'Department': department,
        'JobSatisfaction': job_satisfaction,
        'Overtime': overtime,
        'YearsAtCompany': years_at_company,
        'WorkLifeBalance': work_life_balance,
        'DistanceFromHome': distance_from_home
    })
    
    log_odds = (
        4.2 
        - 0.04 * df['Age'] 
        - 0.000025 * df['Salary'] 
        - 0.4 * df['JobSatisfaction'] 
        + 1.3 * df['Overtime'] 
        - 0.05 * df['YearsAtCompany'] 
        - 0.4 * df['WorkLifeBalance'] 
        + 0.04 * df['DistanceFromHome']
    )
    
    prob = 1 / (1 + np.exp(-log_odds))
    attrition = (np.random.rand(num_rows) < prob).astype(int)
    df['Attrition'] = attrition
    
    return df

def generate_attrition_data_monthly(month, num_rows=1200, seed=42):
    """
    Generates monthly evaluation datasets (Month 1 to 5) with sequential drift:
    - Month 1: Stable baseline reference.
    - Month 2: Overtime rates begin rising (from 30% to 45% probability).
    - Month 3: Gradual salary drop starts (hiring freeze of seniors, junior hires rise).
    - Month 4: Major salary shift (restructuring: median salary drops from $85k to $62k).
    - Month 5: Concept drift: Attrition rises due to high overtime combined with lower salary.
    """
    np.random.seed(seed + month)
    
    age = np.random.normal(37, 9, size=num_rows).astype(int)
    age = np.clip(age, 18, 60)
    
    # Month 3: Salary centers at $76k; Month 4+: centers at $62k
    mean_salary = 85000
    if month == 3:
        mean_salary = 76000
    elif month >= 4:
        mean_salary = 62000
        
    salary = np.random.normal(mean_salary, 20000, size=num_rows).astype(int)
    salary = np.clip(salary, 25000, 200000)
    
    department = np.random.choice([0, 1, 2], size=num_rows, p=[0.30, 0.60, 0.10])
    job_satisfaction = np.random.choice([1, 2, 3, 4], size=num_rows, p=[0.15, 0.25, 0.35, 0.25])
    
    # Month 2+: Higher Overtime
    p_overtime = 0.30
    if month >= 2:
        p_overtime = 0.45
    overtime = np.random.choice([0, 1], size=num_rows, p=[1 - p_overtime, p_overtime])
    
    max_years = age - 18
    years_at_company = np.zeros(num_rows)
    for i in range(num_rows):
        if max_years[i] <= 1:
            years_at_company[i] = 1
        else:
            years_at_company[i] = np.random.choice(range(1, min(int(max_years[i]), 21)))
    years_at_company = years_at_company.astype(int)
    
    work_life_balance = np.random.choice([1, 2, 3, 4], size=num_rows, p=[0.10, 0.25, 0.45, 0.20])
    distance_from_home = np.random.exponential(scale=8.0, size=num_rows).astype(int)
    distance_from_home = np.clip(distance_from_home, 1, 30)
    
    df = pd.DataFrame({
        'Age': age,
        'Salary': salary,
        'Department': department,
        'JobSatisfaction': job_satisfaction,
        'Overtime': overtime,
        'YearsAtCompany': years_at_company,
        'WorkLifeBalance': work_life_balance,
        'DistanceFromHome': distance_from_home
    })
    
    # Month 5+: Concept drift (burnout shifts attrition patterns)
    overtime_coeff = 1.3
    salary_coeff = -0.000025
    intercept = 4.2
    if month >= 5:
        overtime_coeff = 2.2
        salary_coeff = -0.000035
        intercept = 5.0
        
    log_odds = (
        intercept 
        - 0.04 * df['Age'] 
        + salary_coeff * df['Salary'] 
        - 0.4 * df['JobSatisfaction'] 
        + overtime_coeff * df['Overtime'] 
        - 0.05 * df['YearsAtCompany'] 
        - 0.4 * df['WorkLifeBalance'] 
        + 0.04 * df['DistanceFromHome']
    )
    
    prob = 1 / (1 + np.exp(-log_odds))
    attrition = (np.random.rand(num_rows) < prob).astype(int)
    df['Attrition'] = attrition
    
    return df

def main():
    print("=== Generating Datasets for Employee Attrition Prediction ===")
    # Generate main training dataset
    dataset = generate_attrition_data(1500, seed=42)
    dataset.to_csv("dataset.csv", index=False)
    print(f"Saved dataset.csv ({len(dataset)} rows)")
    
    # Generate monthly sequential batches
    for m in range(1, 6):
        month_data = generate_attrition_data_monthly(month=m, num_rows=1200, seed=100 + m)
        month_data.to_csv(f"month_{m}.csv", index=False)
        print(f"Saved month_{m}.csv ({len(month_data)} rows)")
        
        # Save Month 1 as baseline.csv and Month 5 as current.csv for backward compatibility
        if m == 1:
            month_data.to_csv("baseline.csv", index=False)
            print("Copied month_1.csv to baseline.csv")
        elif m == 5:
            month_data.to_csv("current.csv", index=False)
            print("Copied month_5.csv to current.csv")
            
    # Prepare features and target
    X = dataset.drop(columns=['Attrition'])
    y = dataset['Attrition']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    print("\nTraining RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print("\nEvaluation Metrics on Test Split:")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
    
    # Save model
    model_path = "model.pkl"
    joblib.dump(model, model_path)
    print(f"\nModel saved successfully as: {model_path}")
    
    # Verify loaded model compatibility
    print("\nVerifying loaded model compatibility...")
    loaded_model = joblib.load(model_path)
    assert hasattr(loaded_model, "feature_names_in_"), "Model does not preserve feature_names_in_!"
    print("[OK] Model contains feature_names_in_ attribute.")
    
    # Run prediction check on all monthly batches
    for m in range(1, 6):
        df_month = pd.read_csv(f"month_{m}.csv")
        pred = loaded_model.predict(df_month.drop(columns=['Attrition']))
        acc = accuracy_score(df_month['Attrition'], pred)
        print(f"[OK] Predicted successfully on month_{m}.csv (Accuracy: {acc:.4f})")
        
    print("\n=== Employee Attrition Demo Assets Generated Successfully ===")

if __name__ == '__main__':
    main()
