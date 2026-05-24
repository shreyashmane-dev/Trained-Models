import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

def generate_fraud_data(num_rows, seed=42):
    """
    Generates training synthetic insurance claim dataset with baseline features.
    """
    np.random.seed(seed)
    
    claim_amount = np.random.lognormal(mean=8.5, sigma=0.8, size=num_rows).astype(int)
    claim_amount = np.clip(claim_amount, 500, 75000)
    
    customer_age = np.random.normal(45, 14, size=num_rows).astype(int)
    customer_age = np.clip(customer_age, 18, 80)
    
    max_policy_years = customer_age - 18
    policy_years = np.zeros(num_rows)
    for i in range(num_rows):
        if max_policy_years[i] <= 1:
            policy_years[i] = 1
        else:
            policy_years[i] = np.random.choice(range(1, min(int(max_policy_years[i]), 16)))
    policy_years = policy_years.astype(int)
    
    accident_severity = np.random.choice([0, 1, 2], size=num_rows, p=[0.50, 0.35, 0.15])
    number_of_claims = np.random.choice([1, 2, 3, 4, 5], size=num_rows, p=[0.60, 0.22, 0.11, 0.05, 0.02])
    
    vehicle_age = np.random.normal(7, 4, size=num_rows).astype(int)
    vehicle_age = np.clip(vehicle_age, 1, 25)
    
    premium_base = 1200 + 150 * number_of_claims + 30 * vehicle_age
    premium_amount = (premium_base + np.random.normal(0, 200, size=num_rows)).astype(int)
    premium_amount = np.clip(premium_amount, 500, 4000)
    
    df = pd.DataFrame({
        'ClaimAmount': claim_amount,
        'CustomerAge': customer_age,
        'PolicyYears': policy_years,
        'AccidentSeverity': accident_severity,
        'NumberOfClaims': number_of_claims,
        'VehicleAge': vehicle_age,
        'PremiumAmount': premium_amount
    })
    
    log_odds = (
        -1.5
        + 0.00006 * df['ClaimAmount']
        - 0.02 * df['CustomerAge']
        - 0.22 * df['PolicyYears']
        + 1.3 * df['AccidentSeverity']
        + 0.45 * df['NumberOfClaims']
        + 0.03 * df['VehicleAge']
        - 0.0003 * df['PremiumAmount']
    )
    
    prob = 1 / (1 + np.exp(-log_odds))
    fraudulent = (np.random.rand(num_rows) < prob).astype(int)
    df['Fraudulent'] = fraudulent
    
    return df

def generate_fraud_data_monthly(month, num_rows=1200, seed=42):
    """
    Generates monthly evaluation datasets (Month 1 to 5) with sequential drift:
    - Month 1 & 2: Stable claim counts and fraud rates.
    - Month 3: VehicleAge distribution shifts older (mean shifts from 7 to 11).
    - Month 4: ClaimAmount spike due to a simulated catastrophic weather event (mean log-normal parameter shifts from 8.5 to 9.2).
    - Month 5: Concept drift: Fraud ring activity changes (AccidentSeverity becomes highly predictive, ClaimAmount coefficient declines).
    """
    np.random.seed(seed + month)
    
    # Month 4+: Weather event claim amount spike
    mean_log_claim = 8.5
    if month >= 4:
        mean_log_claim = 9.2
    claim_amount = np.random.lognormal(mean=mean_log_claim, sigma=0.8, size=num_rows).astype(int)
    claim_amount = np.clip(claim_amount, 500, 75000)
    
    customer_age = np.random.normal(45, 14, size=num_rows).astype(int)
    customer_age = np.clip(customer_age, 18, 80)
    
    max_policy_years = customer_age - 18
    policy_years = np.zeros(num_rows)
    for i in range(num_rows):
        if max_policy_years[i] <= 1:
            policy_years[i] = 1
        else:
            policy_years[i] = np.random.choice(range(1, min(int(max_policy_years[i]), 16)))
    policy_years = policy_years.astype(int)
    
    accident_severity = np.random.choice([0, 1, 2], size=num_rows, p=[0.50, 0.35, 0.15])
    number_of_claims = np.random.choice([1, 2, 3, 4, 5], size=num_rows, p=[0.60, 0.22, 0.11, 0.05, 0.02])
    
    # Month 3+: Older vehicles insured portfolio
    mean_v_age = 7
    if month >= 3:
        mean_v_age = 11
    vehicle_age = np.random.normal(mean_v_age, 4, size=num_rows).astype(int)
    vehicle_age = np.clip(vehicle_age, 1, 25)
    
    premium_base = 1200 + 150 * number_of_claims + 30 * vehicle_age
    premium_amount = (premium_base + np.random.normal(0, 200, size=num_rows)).astype(int)
    premium_amount = np.clip(premium_amount, 500, 4000)
    
    df = pd.DataFrame({
        'ClaimAmount': claim_amount,
        'CustomerAge': customer_age,
        'PolicyYears': policy_years,
        'AccidentSeverity': accident_severity,
        'NumberOfClaims': number_of_claims,
        'VehicleAge': vehicle_age,
        'PremiumAmount': premium_amount
    })
    
    # Month 5+: Fraud ring patterns shift (Concept Drift)
    claim_coeff = 0.00006
    severity_coeff = 1.3
    intercept = -1.5
    if month >= 5:
        claim_coeff = 0.00003
        severity_coeff = 2.0
        intercept = -2.0
        
    log_odds = (
        intercept
        + claim_coeff * df['ClaimAmount']
        - 0.02 * df['CustomerAge']
        - 0.22 * df['PolicyYears']
        + severity_coeff * df['AccidentSeverity']
        + 0.45 * df['NumberOfClaims']
        + 0.03 * df['VehicleAge']
        - 0.0003 * df['PremiumAmount']
    )
    
    prob = 1 / (1 + np.exp(-log_odds))
    fraudulent = (np.random.rand(num_rows) < prob).astype(int)
    df['Fraudulent'] = fraudulent
    
    return df

def main():
    print("=== Generating Datasets for Insurance Fraud Detection ===")
    # Generate main training dataset
    dataset = generate_fraud_data(1500, seed=42)
    dataset.to_csv("dataset.csv", index=False)
    print(f"Saved dataset.csv ({len(dataset)} rows)")
    
    # Generate monthly sequential batches
    for m in range(1, 6):
        month_data = generate_fraud_data_monthly(month=m, num_rows=1200, seed=100 + m)
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
    X = dataset.drop(columns=['Fraudulent'])
    y = dataset['Fraudulent']
    
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
        pred = loaded_model.predict(df_month.drop(columns=['Fraudulent']))
        acc = accuracy_score(df_month['Fraudulent'], pred)
        print(f"[OK] Predicted successfully on month_{m}.csv (Accuracy: {acc:.4f})")
        
    print("\n=== Insurance Fraud Demo Assets Generated Successfully ===")

if __name__ == '__main__':
    main()
