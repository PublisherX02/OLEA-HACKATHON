"""
Phase VIII v2 - Optuna Feature Selection Search
Stratégie : Laisser l'IA décider quelles features ajouter, pas toutes en bloc.
Optuna explore les combinaisons de features ET les hyperparamètres simultanément.
Baseline à battre : Macro F1 = 0.6307
"""
import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings("ignore")

BASELINE_F1 = 0.6307
print("🤖 Phase VIII v2 - Optuna Feature + HyperParam Co-Search")
print(f"📊 Baseline à battre : {BASELINE_F1}")

df = pd.read_csv("train.csv")
X_raw = df.drop(columns=['Purchased_Coverage_Bundle', 'User_ID'])
y = df['Purchased_Coverage_Bundle']

MONTH_MAP = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12
}

def build_features(df_in, params):
    """Construit le dataset avec seulement les features selectionnées par Optuna."""
    X = df_in.copy()

    # --- Features de base (Phase IV - toujours incluses) ---
    X['Total_Dependents'] = X['Adult_Dependents'] + X['Child_Dependents'].fillna(0) + X['Infant_Dependents']
    X['Income_per_Dependent'] = X['Estimated_Annual_Income'] / (X['Total_Dependents'] + 1)
    X['Risk_Ratio'] = X['Previous_Claims_Filed'] / (X['Years_Without_Claims'] + 1)
    X['Vehicles_per_Adult'] = X['Vehicles_on_Policy'] / (X['Adult_Dependents'] + 1)

    # --- Features Optionnelles (Optuna décide OUI/NON pour chacune) ---
    if params.get('use_temporal', False):
        week = X['Policy_Start_Week'].fillna(26)
        X['Policy_Week_Sin'] = np.sin(2 * np.pi * week / 52)
        X['Policy_Week_Cos'] = np.cos(2 * np.pi * week / 52)
        month_num = X['Policy_Start_Month'].map(MONTH_MAP).fillna(6)
        X['Policy_Month_Sin'] = np.sin(2 * np.pi * month_num / 12)
        X['Policy_Month_Cos'] = np.cos(2 * np.pi * month_num / 12)

    if params.get('use_claims_income', False):
        X['Claims_x_Income'] = X['Previous_Claims_Filed'] * X['Estimated_Annual_Income']

    if params.get('use_risk_vehicles', False):
        X['Risk_x_Vehicles'] = X['Risk_Ratio'] * X['Vehicles_on_Policy']

    if params.get('use_claims_per_vehicle', False):
        X['Claims_Per_Vehicle'] = X['Previous_Claims_Filed'] / (X['Vehicles_on_Policy'] + 1)

    if params.get('use_income_risk_combined', False):
        X['Income_Risk_Combined'] = X['Income_per_Dependent'] * X['Risk_Ratio']

    if params.get('use_direct_buyer', False):
        X['Is_Direct_Buyer'] = X['Broker_ID'].isna().astype(int)

    if params.get('use_zero_ded', False):
        X['Is_Zero_Deductible'] = (X['Deductible_Tier'] == 'Tier_4_Zero_Ded').astype(int)

    if params.get('use_full_time', False):
        X['Is_Full_Time'] = (X['Employment_Status'] == 'Employed_FullTime').astype(int)

    if params.get('use_grace', False):
        X['Has_Grace_Extensions'] = (X['Grace_Period_Extensions'] > 0).astype(int)

    if params.get('use_policy_duration', False):
        X['Policy_Duration_Bucket'] = pd.cut(
            X['Previous_Policy_Duration_Months'].fillna(0),
            bins=[0, 6, 12, 24, 999],
            labels=[0, 1, 2, 3]
        ).astype(float)

    return X

def objective(trial):
    feature_params = {
        'use_temporal':           trial.suggest_categorical('use_temporal', [True, False]),
        'use_claims_income':      trial.suggest_categorical('use_claims_income', [True, False]),
        'use_risk_vehicles':      trial.suggest_categorical('use_risk_vehicles', [True, False]),
        'use_claims_per_vehicle': trial.suggest_categorical('use_claims_per_vehicle', [True, False]),
        'use_income_risk_combined': trial.suggest_categorical('use_income_risk_combined', [True, False]),
        'use_direct_buyer':       trial.suggest_categorical('use_direct_buyer', [True, False]),
        'use_zero_ded':           trial.suggest_categorical('use_zero_ded', [True, False]),
        'use_full_time':          trial.suggest_categorical('use_full_time', [True, False]),
        'use_grace':              trial.suggest_categorical('use_grace', [True, False]),
        'use_policy_duration':    trial.suggest_categorical('use_policy_duration', [True, False]),
    }

    # Hyperparamètres co-optimisés
    model_params = {
        'n_estimators':     trial.suggest_int('n_estimators', 200, 400),
        'learning_rate':    trial.suggest_float('learning_rate', 0.05, 0.15, log=True),
        'num_leaves':       trial.suggest_int('num_leaves', 50, 100),
        'max_depth':        trial.suggest_int('max_depth', 8, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 15, 50),
        'subsample':        trial.suggest_float('subsample', 0.75, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.75, 1.0),
        'reg_alpha':        trial.suggest_float('reg_alpha', 0.05, 0.5, log=True),
        'reg_lambda':       trial.suggest_float('reg_lambda', 0.0005, 0.05, log=True),
        'random_state':     42,
        'class_weight':     'balanced',
        'verbose':          -1,
    }

    X = build_features(X_raw, feature_params)
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_cols = X.select_dtypes(exclude=['object']).columns.tolist()

    preprocessor = ColumnTransformer(transformers=[
        ('num', SimpleImputer(strategy='median'), num_cols),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ]), cat_cols)
    ])

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LGBMClassifier(**model_params))
    ])

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in cv.split(X, y):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[val_idx])
        scores.append(f1_score(y.iloc[val_idx], preds, average='macro'))

    return np.mean(scores)

# Callback auto-stop si on dépasse 65%
def stop_at_65(study, trial):
    if study.best_value >= 0.65:
        print(f"\n🛑 OBJECTIF 65% ATTEINT : {study.best_value:.4f}. Arrêt.")
        study.stop()

study = optuna.create_study(direction="maximize", study_name="Feature_Selection_v2")
print("\n🚀 Lancement de la recherche sur 60 combinaisons...")
study.optimize(objective, n_trials=60, callbacks=[stop_at_65])

new_f1 = study.best_value
best = study.best_params

print("\n" + "=" * 60)
print(f"🏆 MEILLEUR RÉSULTAT : Macro F1 = {new_f1:.4f}")
print(f"📊 Baseline précédente           = {BASELINE_F1:.4f}")
print()

if new_f1 > BASELINE_F1:
    gain = (new_f1 - BASELINE_F1) * 100
    print(f"✅ AMÉLIORATION : +{gain:.2f}% !")
    print("⚙️  Paramètres gagnants :")
    for k, v in best.items():
        print(f"    {k} = {v}")
else:
    print(f"❌ Pas d'amélioration. Baseline conservée.")
