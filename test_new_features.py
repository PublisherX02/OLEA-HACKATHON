"""
Phase VIII - Feature Engineering Safety Test
Ce script teste les nouvelles features sur un CV 5-Fold.
Il ne modifie AUCUN fichier de production (model.pkl, submission.zip, build_model.py).
Décision : Si Macro F1 > 0.6307, déploiement. Sinon, abandon.
"""
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

BASELINE_F1 = 0.6307   # Notre champion actuel Phase VII

print("🧪 Phase VIII - Test Sécurisé des Nouvelles Features")
print("=" * 55)
print(f"📊 Baseline à battre : Macro F1 = {BASELINE_F1}")
print()

df = pd.read_csv("train.csv")
X = df.drop(columns=['Purchased_Coverage_Bundle', 'User_ID'])
y = df['Purchased_Coverage_Bundle']

# =========================================================
# BLOCK 1 : Features de base (Phase IV - déjà validées)
# =========================================================
def apply_base_features(df):
    X_fe = df.copy()
    X_fe['Total_Dependents'] = X_fe['Adult_Dependents'] + X_fe['Child_Dependents'].fillna(0) + X_fe['Infant_Dependents']
    X_fe['Income_per_Dependent'] = X_fe['Estimated_Annual_Income'] / (X_fe['Total_Dependents'] + 1)
    X_fe['Risk_Ratio'] = X_fe['Previous_Claims_Filed'] / (X_fe['Years_Without_Claims'] + 1)
    X_fe['Vehicles_per_Adult'] = X_fe['Vehicles_on_Policy'] / (X_fe['Adult_Dependents'] + 1)
    return X_fe

# =========================================================
# BLOCK 2 : Nouvelles features Phase VIII
# =========================================================
MONTH_MAP = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12
}

def apply_new_features(df):
    X_fe = df.copy()

    # --- Signaux Temporels Cycliques ---
    week = X_fe['Policy_Start_Week'].fillna(26)
    X_fe['Policy_Week_Sin'] = np.sin(2 * np.pi * week / 52)
    X_fe['Policy_Week_Cos'] = np.cos(2 * np.pi * week / 52)

    month_num = X_fe['Policy_Start_Month'].map(MONTH_MAP).fillna(6)
    X_fe['Policy_Month_Sin'] = np.sin(2 * np.pi * month_num / 12)
    X_fe['Policy_Month_Cos'] = np.cos(2 * np.pi * month_num / 12)

    # --- Features d'Interaction ---
    X_fe['Claims_x_Income'] = X_fe['Previous_Claims_Filed'] * X_fe['Estimated_Annual_Income']
    X_fe['Risk_x_Vehicles'] = X_fe['Risk_Ratio'] * X_fe['Vehicles_on_Policy']
    X_fe['Claims_Per_Vehicle'] = X_fe['Previous_Claims_Filed'] / (X_fe['Vehicles_on_Policy'] + 1)
    X_fe['Income_Risk_Combined'] = X_fe['Income_per_Dependent'] * X_fe['Risk_Ratio']

    # --- Signaux Binaires (Drapeaux Métier) ---
    X_fe['Is_Direct_Buyer'] = X_fe['Broker_ID'].isna().astype(int)
    X_fe['Is_Zero_Deductible'] = (X_fe['Deductible_Tier'] == 'Tier_4_Zero_Ded').astype(int)
    X_fe['Is_Full_Time'] = (X_fe['Employment_Status'] == 'Employed_FullTime').astype(int)
    X_fe['Has_Grace_Extensions'] = (X_fe['Grace_Period_Extensions'] > 0).astype(int)

    return X_fe

def apply_all_features(df):
    df = apply_base_features(df)
    df = apply_new_features(df)
    return df

print("⚙️ Application de toutes les features (base + nouvelles)...")
X = apply_all_features(X)

# Colonnes du modèle
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(exclude=['object']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), num_cols),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ]), cat_cols)
    ])

# Hyperparamètres Optuna (Phase VII - champions)
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LGBMClassifier(
        random_state=42,
        class_weight='balanced',
        n_estimators=265,
        learning_rate=0.09801317587872153,
        num_leaves=70,
        max_depth=10,
        min_child_samples=27,
        subsample=0.8203306012855914,
        colsample_bytree=0.9662869673181392,
        reg_alpha=0.12931108690860685,
        reg_lambda=0.0012305235589046113,
        verbose=-1
    ))
])

# 5-Fold Cross-Validation (plus robuste que 3-Fold)
print("🔬 Cross-Validation 5-Fold en cours... (patience ~3 min)")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
f1_scores = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    score = f1_score(y_val, preds, average='macro')
    f1_scores.append(score)
    print(f"  Fold {fold}/5 → F1 Macro = {score:.4f}")

new_f1 = np.mean(f1_scores)

print()
print("=" * 55)
print(f"🏆 RÉSULTAT FINAL : Macro F1 Moyen = {new_f1:.4f}")
print(f"📊 Baseline précédente              = {BASELINE_F1:.4f}")
print()

if new_f1 > BASELINE_F1:
    gain = (new_f1 - BASELINE_F1) * 100
    print(f"✅ AMÉLIORATION CONFIRMÉE : +{gain:.2f}% de gain !")
    print("🚀 DÉCISION : Déploiement des nouvelles features approuvé.")
    print("   → Lance build_model.py pour regenerer model.pkl + submission.zip")
else:
    drop = (BASELINE_F1 - new_f1) * 100
    print(f"❌ RÉGRESSION : -{drop:.2f}% de chute. Abandon immédiat.")
    print("   → Aucun fichier de production n'a été modifié.")
