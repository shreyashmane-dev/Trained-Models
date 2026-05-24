import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

def generate_loan_data(num_rows, seed=42):
    """
    Generates training synthetic loan approval dataset with baseline features.
    """
    np.random.seed(seed)
    
    gender = np.random.choice([0, 1], size=num_rows, p=[0.40, 0.60])
    married = np.random.choice([0, 1], size=num_rows, p=[0.35, 0.65])
    dependents = np.random.choice([0, 1, 2, 3], size=num_rows, p=[0.55, 0.18, 0.15, 0.12])
    education = np.random.choice([0, 1], size=num_rows, p=[0.22, 0.78])
    
    applicant_income = np.random.lognormal(mean=8.4, sigma=0.5, size=num_rows).astype(int)
    applicant_income = np.clip(applicant_income, 1500, 25000)
    
    has_coapplicant = np.random.choice([0, 1], size=num_rows, p=[0.45, 0.55])
    coapplicant_income = np.zeros(num_rows)
    co_income_values = np.random.lognormal(mean=7.6, sigma=0.6, size=num_rows).astype(int)
    co_income_values = np.clip(co_income_values, 1000, 15000)
    coapplicant_income[has_coapplicant == 1] = co_income_values[has_coapplicant == 1]
    
    total_income = applicant_income + coapplicant_income
    loan_amount = (0.02 * total_income + np.random.normal(50, 25, size=num_rows)).astype(int)
    loan_amount = np.clip(loan_amount, 30, 600)
    
    credit_history = np.random.choice([0, 1], size=num_rows, p=[0.16, 0.84])
    property_area = np.random.choice([0, 1, 2], size=num_rows, p=[0.30, 0.38, 0.32])
    
    df = pd.DataFrame({
        'Gender': gender,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'CreditHistory': credit_history,
        'PropertyArea': property_area
    })
    
    annual_income_k = (df['ApplicantIncome'] + df['CoapplicantIncome']) * 12 / 1000.0
    loan_to_income = df['LoanAmount'] / (annual_income_k + 1e-5)
    
    log_odds = (
        -1.5 
        + 3.5 * df['CreditHistory'] 
        + 0.6 * df['Education'] 
        + 0.4 * df['Married']
        - 0.8 * loan_to_income
        + 0.8 * df['Gender']
        + 0.2 * (df['PropertyArea'] == 1).astype(int)
    )
    
    prob = 1 / (1 + np.exp(-log_odds))
    loan_status = (np.random.rand(num_rows) < prob).astype(int)
    df['Loan_Status'] = loan_status
    
    return df

def generate_loan_data_monthly(month, num_rows=1200, seed=42):
    """
    Generates monthly evaluation datasets (Month 1 to 5) with sequential drift:
    - Month 1 & 2: Stable, matching baseline.
    - Month 3: Gradual rise in ApplicantIncome (higher median due to inflation).
    - Month 4: Recesssion hits: CreditHistory rate declines (fewer good history).
    - Month 5: Concept drift: Underwriting rules change (CreditHistory becomes less predictive, Education more predictive).
    """
    np.random.seed(seed + month)
    
    gender = np.random.choice([0, 1], size=num_rows, p=[0.40, 0.60])
    married = np.random.choice([0, 1], size=num_rows, p=[0.35, 0.65])
    dependents = np.random.choice([0, 1, 2, 3], size=num_rows, p=[0.55, 0.18, 0.15, 0.12])
    education = np.random.choice([0, 1], size=num_rows, p=[0.22, 0.78])
    
    # Month 3+: Inflation drift on income
    mean_income = 8.4
    if month >= 3:
        mean_income = 8.7
        
    applicant_income = np.random.lognormal(mean=mean_income, sigma=0.5, size=num_rows).astype(int)
    applicant_income = np.clip(applicant_income, 1500, 25000)
    
    has_coapplicant = np.random.choice([0, 1], size=num_rows, p=[0.45, 0.55])
    coapplicant_income = np.zeros(num_rows)
    co_income_values = np.random.lognormal(mean=7.6, sigma=0.6, size=num_rows).astype(int)
    co_income_values = np.clip(co_income_values, 1000, 15000)
    coapplicant_income[has_coapplicant == 1] = co_income_values[has_coapplicant == 1]
    
    total_income = applicant_income + coapplicant_income
    loan_amount = (0.02 * total_income + np.random.normal(50, 25, size=num_rows)).astype(int)
    loan_amount = np.clip(loan_amount, 30, 600)
    
    # Month 4+: Recession drift on credit history
    p_good_credit = 0.84
    if month >= 4:
        p_good_credit = 0.55
        
    credit_history = np.random.choice([0, 1], size=num_rows, p=[1 - p_good_credit, p_good_credit])
    property_area = np.random.choice([0, 1, 2], size=num_rows, p=[0.30, 0.38, 0.32])
    
    df = pd.DataFrame({
        'Gender': gender,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'CreditHistory': credit_history,
        'PropertyArea': property_area
    })
    
    annual_income_k = (df['ApplicantIncome'] + df['CoapplicantIncome']) * 12 / 1000.0
    loan_to_income = df['LoanAmount'] / (annual_income_k + 1e-5)
    
    # Month 5+: Concept drift on underwriting rules
    credit_coeff = 3.5
    edu_coeff = 0.6
    if month >= 5:
        credit_coeff = 2.0
        edu_coeff = 1.5
        
    log_odds = (
        -1.5 
        + credit_coeff * df['CreditHistory'] 
        + edu_coeff * df['Education'] 
        + 0.4 * df['Married']
        - 0.8 * loan_to_income
        + 0.8 * df['Gender']
        + 0.2 * (df['PropertyArea'] == 1).astype(int)
    )
    
    prob = 1 / (1 + np.exp(-log_odds))
    loan_status = (np.random.rand(num_rows) < prob).astype(int)
    df['Loan_Status'] = loan_status
    
    return df

def main():
    print("=== Generating Datasets for Loan Approval Prediction ===")
    # Generate main training dataset
    dataset = generate_loan_data(1500, seed=42)
    dataset.to_csv("dataset.csv", index=False)
    print(f"Saved dataset.csv ({len(dataset)} rows)")
    
    # Generate monthly sequential batches
    for m in range(1, 6):
        month_data = generate_loan_data_monthly(month=m, num_rows=1200, seed=100 + m)
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
    X = dataset.drop(columns=['Loan_Status'])
    y = dataset['Loan_Status']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    print("\nTraining RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
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
        pred = loaded_model.predict(df_month.drop(columns=['Loan_Status']))
        acc = accuracy_score(df_month['Loan_Status'], pred)
        print(f"[OK] Predicted successfully on month_{m}.csv (Accuracy: {acc:.4f})")
        
    print("\n=== Loan Approval Demo Assets Generated Successfully ===")

if __name__ == '__main__':
    main()
