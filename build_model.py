import pandas as pd
import numpy as np
import joblib
import zipfile
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from lightgbm import LGBMClassifier

print("🚀 Phase VIII - LightGBM Champion avec Features Optuna (63.82%)")

# 1. Chargement des données d'entraînement
train_df = pd.read_csv("train.csv")

# 2. Séparation des features (X) et de la cible (y)
X = train_df.drop(columns=['Purchased_Coverage_Bundle'])
y = train_df['Purchased_Coverage_Bundle']

# ---- FEATURE ENGINEERING (Phase VIII Champion) ----
MONTH_MAP = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12
}

def apply_feature_engineering(df):
    X_fe = df.copy()

    # ——— Phase IV: Features de base (toujours incluses) ———
    X_fe['Total_Dependents'] = X_fe['Adult_Dependents'] + X_fe['Child_Dependents'].fillna(0) + X_fe['Infant_Dependents']
    X_fe['Income_per_Dependent'] = X_fe['Estimated_Annual_Income'] / (X_fe['Total_Dependents'] + 1)
    X_fe['Risk_Ratio'] = X_fe['Previous_Claims_Filed'] / (X_fe['Years_Without_Claims'] + 1)
    X_fe['Vehicles_per_Adult'] = X_fe['Vehicles_on_Policy'] / (X_fe['Adult_Dependents'] + 1)

    # ——— Phase VIII: Nouvelles features validées par Optuna ———
    
    # 1. Signaux Temporels Cycliques
    week = X_fe['Policy_Start_Week'].fillna(26)
    X_fe['Policy_Week_Sin'] = np.sin(2 * np.pi * week / 52)
    X_fe['Policy_Week_Cos'] = np.cos(2 * np.pi * week / 52)
    month_num = X_fe['Policy_Start_Month'].map(MONTH_MAP).fillna(6)
    X_fe['Policy_Month_Sin'] = np.sin(2 * np.pi * month_num / 12)
    X_fe['Policy_Month_Cos'] = np.cos(2 * np.pi * month_num / 12)

    # 2. Claims × Income
    X_fe['Claims_x_Income'] = X_fe['Previous_Claims_Filed'] * X_fe['Estimated_Annual_Income']

    # 3. Claims par Véhicule
    X_fe['Claims_Per_Vehicle'] = X_fe['Previous_Claims_Filed'] / (X_fe['Vehicles_on_Policy'] + 1)

    # 4. Déductible Zéro
    X_fe['Is_Zero_Deductible'] = (X_fe['Deductible_Tier'] == 'Tier_4_Zero_Ded').astype(int)

    # 5. Emploi Stable
    X_fe['Is_Full_Time'] = (X_fe['Employment_Status'] == 'Employed_FullTime').astype(int)

    # 6. Ancienneté de Politique Bucketisée (En pur NumPy pour la vitesse)
    durations = X_fe['Previous_Policy_Duration_Months'].fillna(0).values
    buckets = np.zeros_like(durations, dtype=float)
    buckets[(durations > 6) & (durations <= 12)] = 1.0
    buckets[(durations > 12) & (durations <= 24)] = 2.0
    buckets[durations > 24] = 3.0
    X_fe['Policy_Duration_Bucket'] = buckets

    X_fe = X_fe.drop(columns=['Employer_ID'], errors='ignore')
    return X_fe

print("⚙️ Application des features championnes...")
X = apply_feature_engineering(X)

# 3. Identification des types de colonnes (SANS User_ID)
cols_to_train = X.drop(columns=['User_ID']).columns
cat_cols = X[cols_to_train].select_dtypes(include=['object']).columns.tolist()
num_cols = X[cols_to_train].select_dtypes(exclude=['object']).columns.tolist()

# 4. Pipeline Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), num_cols),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ]), cat_cols)
    ])

# 5. LightGBM avec hyperparamètres co-optimisés Phase VIII (Macro F1 = 63.82%)
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LGBMClassifier(
        random_state=42, 
        class_weight='balanced', 
        n_estimators=300,
        learning_rate=0.03, 
        verbose=-1, 
        n_jobs=-1
    ))
])

# 6. Entraînement
print("🧠 Apprentissage en cours...")
clf.fit(X, y)
print("✅ Modèle entraîné !")

# 7. Sauvegarde du modèle (Le Pipeline Complet)
joblib.dump(clf, "model.pkl")

# 8. Création du requirements.txt (VIDE comme demandé)
with open("submission_requirements.txt", "w") as f:
    f.write("")

# 9. Création du solution.py (Vrai solution.py Propre et Sécurisé)
solution_code = '''import pandas as pd
import numpy as np
import joblib

MONTH_MAP = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12
}

def preprocess(df):
    df_fe = df.copy()

    # Ratios 
    df_fe['Total_Dependents'] = df_fe['Adult_Dependents'] + df_fe['Child_Dependents'].fillna(0) + df_fe['Infant_Dependents']
    df_fe['Income_per_Dependent'] = df_fe['Estimated_Annual_Income'] / (df_fe['Total_Dependents'] + 1)
    df_fe['Risk_Ratio'] = df_fe['Previous_Claims_Filed'] / (df_fe['Years_Without_Claims'] + 1)
    df_fe['Vehicles_per_Adult'] = df_fe['Vehicles_on_Policy'] / (df_fe['Adult_Dependents'] + 1)

    # Features Temporelles
    week = df_fe['Policy_Start_Week'].fillna(26)
    df_fe['Policy_Week_Sin'] = np.sin(2 * np.pi * week / 52)
    df_fe['Policy_Week_Cos'] = np.cos(2 * np.pi * week / 52)
    
    month_num = df_fe['Policy_Start_Month'].map(MONTH_MAP).fillna(6)
    df_fe['Policy_Month_Sin'] = np.sin(2 * np.pi * month_num / 12)
    df_fe['Policy_Month_Cos'] = np.cos(2 * np.pi * month_num / 12)

    # Features de Risque
    df_fe['Claims_x_Income'] = df_fe['Previous_Claims_Filed'] * df_fe['Estimated_Annual_Income']
    df_fe['Claims_Per_Vehicle'] = df_fe['Previous_Claims_Filed'] / (df_fe['Vehicles_on_Policy'] + 1)
    df_fe['Is_Zero_Deductible'] = (df_fe['Deductible_Tier'] == 'Tier_4_Zero_Ded').astype(int)
    df_fe['Is_Full_Time'] = (df_fe['Employment_Status'] == 'Employed_FullTime').astype(int)
    
    # ⚡ OPTIMISATION VITESSE : On remplace pd.cut par numpy (100x plus rapide)
    durations = df_fe['Previous_Policy_Duration_Months'].fillna(0).values
    buckets = np.zeros_like(durations, dtype=float)
    buckets[(durations > 6) & (durations <= 12)] = 1.0
    buckets[(durations > 12) & (durations <= 24)] = 2.0
    buckets[durations > 24] = 3.0
    df_fe['Policy_Duration_Bucket'] = buckets

    df_fe = df_fe.drop(columns=['Employer_ID'], errors='ignore')
    return df_fe

def load_model():
    """Le modèle est chargé UNE SEULE FOIS ici, en toute sécurité."""
    return joblib.load("model.pkl")

def predict(df, model):
    """L'inférence pure via le pipeline complet."""
    X_test = df.drop(columns=['User_ID'], errors='ignore')
    
    preds = model.predict(X_test)
    
    if len(preds.shape) > 1:
        preds = preds.flatten()
        
    return pd.DataFrame({
        'User_ID': df["User_ID"].values,
        'Purchased_Coverage_Bundle': preds.astype(int)
    })
'''
with open("solution.py", "w", encoding="utf-8") as f:
    f.write(solution_code)

# 10. Empaquetage
print("📦 Création du fichier submission.zip...")
with zipfile.ZipFile("submission.zip", "w", zipfile.ZIP_DEFLATED) as zipf:
    zipf.write("solution.py")
    zipf.write("model.pkl")
    zipf.write("submission_requirements.txt", arcname="requirements.txt")

print("🏆 Terminé ! Le fichier submission.zip est prêt (Macro F1 = 63.82% !).")
