import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

def generate_heart_data(num_rows, seed=42):
    """
    Generates training synthetic heart disease patient dataset with baseline features.
    """
    np.random.seed(seed)
    
    age = np.random.normal(54, 9, size=num_rows).astype(int)
    age = np.clip(age, 29, 77)
    sex = np.random.choice([0, 1], size=num_rows, p=[0.35, 0.65])
    chest_pain_type = np.random.choice([0, 1, 2, 3], size=num_rows, p=[0.15, 0.20, 0.25, 0.40])
    
    resting_bp = np.random.normal(130, 17, size=num_rows).astype(int)
    resting_bp = np.clip(resting_bp, 90, 200)
    
    cholesterol = np.random.normal(240, 50, size=num_rows).astype(int)
    cholesterol = np.clip(cholesterol, 100, 550)
    
    fasting_bs = np.random.choice([0, 1], size=num_rows, p=[0.85, 0.15])
    
    max_hr_mean = 200 - 0.7 * age
    max_hr = np.random.normal(max_hr_mean, 15, size=num_rows).astype(int)
    max_hr = np.clip(max_hr, 60, 202)
    
    exercise_angina = np.random.choice([0, 1], size=num_rows, p=[0.60, 0.40])
    
    oldpeak = np.random.exponential(scale=1.0, size=num_rows)
    oldpeak = np.clip(oldpeak, 0.0, 6.2)
    oldpeak = np.round(oldpeak, 1)
    
    df = pd.DataFrame({
        'Age': age,
        'Sex': sex,
        'ChestPainType': chest_pain_type,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'ExerciseAngina': exercise_angina,
        'Oldpeak': oldpeak
    })
    
    log_odds = (
        -5.0
        + 0.03 * df['Age']
        + 0.8 * df['Sex']
        + 1.5 * (df['ChestPainType'] == 3).astype(int)
        + 0.012 * df['RestingBP']
        + 0.003 * df['Cholesterol']
        + 0.6 * df['FastingBS']
        - 0.015 * df['MaxHR']
        + 1.2 * df['ExerciseAngina']
        + 0.8 * df['Oldpeak']
    )
    
    prob = 1 / (1 + np.exp(-log_odds))
    heart_disease = (np.random.rand(num_rows) < prob).astype(int)
    df['HeartDisease'] = heart_disease
    
    return df

def generate_heart_data_monthly(month, num_rows=1200, seed=42):
    """
    Generates monthly evaluation datasets (Month 1 to 5) with sequential drift:
    - Month 1 & 2: Stable clinical baseline.
    - Month 3: Older patient demographic shift (Age mean shifts from 54 to 62).
    - Month 4: FastingBS rate increases (from 15% to 35% probability).
    - Month 5: Concept drift: Diagnosis updates (Oldpeak becomes more predictive, intercept changes).
    """
    np.random.seed(seed + month)
    
    # Month 3+: Older age demographics
    mean_age = 54
    if month >= 3:
        mean_age = 62
    age = np.random.normal(mean_age, 9, size=num_rows).astype(int)
    age = np.clip(age, 29, 77)
    
    sex = np.random.choice([0, 1], size=num_rows, p=[0.35, 0.65])
    chest_pain_type = np.random.choice([0, 1, 2, 3], size=num_rows, p=[0.15, 0.20, 0.25, 0.40])
    
    resting_bp = np.random.normal(130, 17, size=num_rows).astype(int)
    resting_bp = np.clip(resting_bp, 90, 200)
    
    cholesterol = np.random.normal(240, 50, size=num_rows).astype(int)
    cholesterol = np.clip(cholesterol, 100, 550)
    
    # Month 4+: High fasting blood sugar referrals
    p_high_fbs = 0.15
    if month >= 4:
        p_high_fbs = 0.35
    fasting_bs = np.random.choice([0, 1], size=num_rows, p=[1 - p_high_fbs, p_high_fbs])
    
    max_hr_mean = 200 - 0.7 * age
    max_hr = np.random.normal(max_hr_mean, 15, size=num_rows).astype(int)
    max_hr = np.clip(max_hr, 60, 202)
    
    exercise_angina = np.random.choice([0, 1], size=num_rows, p=[0.60, 0.40])
    
    oldpeak = np.random.exponential(scale=1.0, size=num_rows)
    oldpeak = np.clip(oldpeak, 0.0, 6.2)
    oldpeak = np.round(oldpeak, 1)
    
    df = pd.DataFrame({
        'Age': age,
        'Sex': sex,
        'ChestPainType': chest_pain_type,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'ExerciseAngina': exercise_angina,
        'Oldpeak': oldpeak
    })
    
    # Month 5+: Concept drift on diagnostic rules
    oldpeak_coeff = 0.8
    intercept = -5.0
    if month >= 5:
        oldpeak_coeff = 1.4
        intercept = -6.0
        
    log_odds = (
        intercept
        + 0.03 * df['Age']
        + 0.8 * df['Sex']
        + 1.5 * (df['ChestPainType'] == 3).astype(int)
        + 0.012 * df['RestingBP']
        + 0.003 * df['Cholesterol']
        + 0.6 * df['FastingBS']
        - 0.015 * df['MaxHR']
        + 1.2 * df['ExerciseAngina']
        + oldpeak_coeff * df['Oldpeak']
    )
    
    prob = 1 / (1 + np.exp(-log_odds))
    heart_disease = (np.random.rand(num_rows) < prob).astype(int)
    df['HeartDisease'] = heart_disease
    
    return df

def main():
    print("=== Generating Datasets for Heart Disease Prediction ===")
    # Generate main training dataset
    dataset = generate_heart_data(1500, seed=42)
    dataset.to_csv("dataset.csv", index=False)
    print(f"Saved dataset.csv ({len(dataset)} rows)")
    
    # Generate monthly sequential batches
    for m in range(1, 6):
        month_data = generate_heart_data_monthly(month=m, num_rows=1200, seed=100 + m)
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
    X = dataset.drop(columns=['HeartDisease'])
    y = dataset['HeartDisease']
    
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
        pred = loaded_model.predict(df_month.drop(columns=['HeartDisease']))
        acc = accuracy_score(df_month['HeartDisease'], pred)
        print(f"[OK] Predicted successfully on month_{m}.csv (Accuracy: {acc:.4f})")
        
    print("\n=== Heart Disease Demo Assets Generated Successfully ===")

if __name__ == '__main__':
    main()
