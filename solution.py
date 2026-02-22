# ----------------------------------------------------------------
# IMPORTANT: This template will be used to evaluate your solution.
#
# Do NOT change the function signatures.
# And ensure that your code runs within the time limits.
# The time calculation will be computed for the predict function only.
#
# Good luck!
# ----------------------------------------------------------------

import pandas as pd
import numpy as np
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

FEATURE_COLS = [
    'Policy_Cancelled_Post_Purchase', 'Policy_Start_Year', 'Policy_Start_Week',
    'Policy_Start_Day', 'Grace_Period_Extensions', 'Previous_Policy_Duration_Months',
    'Adult_Dependents', 'Child_Dependents', 'Infant_Dependents', 'Region_Code',
    'Existing_Policyholder', 'Previous_Claims_Filed', 'Years_Without_Claims',
    'Policy_Amendments_Count', 'Broker_ID', 'Employer_ID', 'Underwriting_Processing_Days',
    'Vehicles_on_Policy', 'Custom_Riders_Requested', 'Broker_Agency_Type',
    'Deductible_Tier', 'Acquisition_Channel', 'Payment_Schedule', 'Employment_Status',
    'Estimated_Annual_Income', 'Days_Since_Quote', 'Policy_Start_Month',
    'Total_Dependents', 'Has_Family', 'Income_per_Dependent', 'Risk_Score',
    'Is_High_Income', 'Has_Vehicle', 'Claims_per_Year', 'No_Income',
    'Has_Broker', 'Has_Employer', 'Total_Complexity', 'Income_x_Family',
    'Income_x_Vehicle', 'Income_x_Dependents', 'Log_Income', 'Income_Bracket',
    'Vehicle_x_Dependents', 'Claims_Risk'
]


def _engineer_features(df):
    month_map = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4,
        'May': 5, 'June': 6, 'July': 7, 'August': 8,
        'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    df['Policy_Start_Month'] = df['Policy_Start_Month'].map(month_map)
    df['Child_Dependents'] = df['Child_Dependents'].fillna(0)
    df['Broker_ID'] = df['Broker_ID'].fillna(-1)
    df['Employer_ID'] = df['Employer_ID'].fillna(-1)
    for col in ['Region_Code', 'Broker_Agency_Type', 'Deductible_Tier',
                'Acquisition_Channel', 'Payment_Schedule', 'Employment_Status']:
        df[col] = df[col].fillna('Unknown').astype('category').cat.codes
    df['Total_Dependents'] = df['Adult_Dependents'] + df['Child_Dependents'] + df['Infant_Dependents']
    df['Has_Family'] = ((df['Child_Dependents'] + df['Infant_Dependents']) > 0).astype(int)
    df['Log_Income'] = np.log1p(df['Estimated_Annual_Income'])
    df['Income_per_Dependent'] = df['Estimated_Annual_Income'] / (df['Total_Dependents'] + 1)
    df['Is_High_Income'] = (df['Estimated_Annual_Income'] > 50000).astype(int)
    df['No_Income'] = (df['Estimated_Annual_Income'] == 0).astype(int)
    df['Income_Bracket'] = pd.cut(
        df['Estimated_Annual_Income'],
        bins=[-1, 15000, 30000, 50000, 80000, 1e9],
        labels=[0, 1, 2, 3, 4]
    ).astype(float).fillna(0)
    df['Has_Vehicle'] = (df['Vehicles_on_Policy'] > 0).astype(int)
    df['Risk_Score'] = df['Previous_Claims_Filed'] - df['Years_Without_Claims']
    df['Claims_per_Year'] = df['Previous_Claims_Filed'] / (df['Previous_Policy_Duration_Months'] / 12 + 1)
    df['Claims_Risk'] = df['Previous_Claims_Filed'] * (1 + df['Grace_Period_Extensions'])
    df['Has_Broker'] = (df['Broker_ID'] != -1).astype(int)
    df['Has_Employer'] = (df['Employer_ID'] != -1).astype(int)
    df['Total_Complexity'] = (df['Policy_Amendments_Count']
                               + df['Grace_Period_Extensions']
                               + df['Custom_Riders_Requested'])
    df['Income_x_Family'] = df['Estimated_Annual_Income'] * df['Has_Family']
    df['Income_x_Vehicle'] = df['Estimated_Annual_Income'] * df['Has_Vehicle']
    df['Income_x_Dependents'] = df['Estimated_Annual_Income'] * df['Total_Dependents']
    df['Vehicle_x_Dependents'] = df['Vehicles_on_Policy'] * df['Total_Dependents']
    return df


def preprocess(df):
    df = df.copy()
    df = _engineer_features(df)
    return df


def load_model():
    model = joblib.load('model.pkl')
    return model


def predict(df, model):
    X = df[FEATURE_COLS].values.astype(np.float32)
    preds = model.predict(X)

    # Rule-based override for class 9 (Renter_Premium)
    # Only 5 training samples — identified by: income=0, Tier_4_Zero_Ded, no vehicle
    # Tier_4_Zero_Ded encodes to 3 (alphabetical order among categories)
    deductible_idx = FEATURE_COLS.index('Deductible_Tier')
    income_idx = FEATURE_COLS.index('Estimated_Annual_Income')
    vehicle_idx = FEATURE_COLS.index('Vehicles_on_Policy')
    mask_9 = (X[:, income_idx] == 0) & (X[:, deductible_idx] == 3) & (X[:, vehicle_idx] == 0)
    preds[mask_9] = 9

    return pd.DataFrame({
        'User_ID': df['User_ID'].values,
        'Purchased_Coverage_Bundle': preds.astype(int)
    })


# ----------------------------------------------------------------
# TRAINING SCRIPT — run locally: python solution.py --train
# ----------------------------------------------------------------

def _oversample(X, y, min_s=2000):
    from sklearn.utils import resample
    X_list, y_list = [X], [y]
    for cls in range(10):
        mask = y == cls
        cnt = mask.sum()
        if cnt < min_s:
            X_list.append(resample(X[mask], n_samples=min_s - cnt, replace=True, random_state=42))
            y_list.append(pd.Series([cls] * (min_s - cnt)))
    return (pd.concat(X_list).reset_index(drop=True),
            pd.concat(y_list).reset_index(drop=True))


def train():
    print("Loading data...")
    df = pd.read_csv('train.csv')
    df = preprocess(df)
    X = df[FEATURE_COLS]
    y = df['Purchased_Coverage_Bundle']
    print("Oversampling rare classes...")
    X_os, y_os = _oversample(X, y, min_s=2000)
    print("Training model...")
    model = BaggingClassifier(
        estimator=DecisionTreeClassifier(max_depth=None, class_weight='balanced'),
        n_estimators=5, random_state=42, n_jobs=1
    )
    model.fit(X_os.values.astype(np.float32), y_os.values)
    joblib.dump(model, 'model.pkl', compress=3)
    import os
    print(f"Saved model.pkl ({os.path.getsize('model.pkl')/1e6:.1f} MB)")


if __name__ == '__main__':
    import sys
    if '--train' in sys.argv:
        train()
    else:
        print("Usage: python solution.py --train")


def run(df) -> tuple[float, float, float]:
    import time as _time
    df_processed = preprocess(df)
    model = load_model()
    size = get_model_size(model)
    start = _time.perf_counter()
    predictions = predict(df_processed, model)
    duration = _time.perf_counter() - start
    accuracy = get_model_accuracy(predictions)
    return size, accuracy, duration


def get_model_size(model):
    pass


def get_model_accuracy(predictions):
    pass
