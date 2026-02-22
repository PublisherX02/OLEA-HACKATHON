import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from lightgbm import LGBMClassifier

print("📊 Chargement des données et séparation Train/Validation...")
df = pd.read_csv("train.csv")

X = df.drop(columns=['Purchased_Coverage_Bundle', 'User_ID'])
y = df['Purchased_Coverage_Bundle']

# ---- FEATURE ENGINEERING (Phase VIII Champion = 63.82% Macro F1) ----
MONTH_MAP = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12
}

def apply_feature_engineering(df):
    X_fe = df.copy()

    # Phase IV Features (base)
    X_fe['Total_Dependents'] = X_fe['Adult_Dependents'] + X_fe['Child_Dependents'].fillna(0) + X_fe['Infant_Dependents']
    X_fe['Income_per_Dependent'] = X_fe['Estimated_Annual_Income'] / (X_fe['Total_Dependents'] + 1)
    X_fe['Risk_Ratio'] = X_fe['Previous_Claims_Filed'] / (X_fe['Years_Without_Claims'] + 1)
    X_fe['Vehicles_per_Adult'] = X_fe['Vehicles_on_Policy'] / (X_fe['Adult_Dependents'] + 1)

    # Phase VIII Features (validées par Optuna v2)
    week = X_fe['Policy_Start_Week'].fillna(26)
    X_fe['Policy_Week_Sin'] = np.sin(2 * np.pi * week / 52)
    X_fe['Policy_Week_Cos'] = np.cos(2 * np.pi * week / 52)
    month_num = X_fe['Policy_Start_Month'].map(MONTH_MAP).fillna(6)
    X_fe['Policy_Month_Sin'] = np.sin(2 * np.pi * month_num / 12)
    X_fe['Policy_Month_Cos'] = np.cos(2 * np.pi * month_num / 12)

    X_fe['Claims_x_Income'] = X_fe['Previous_Claims_Filed'] * X_fe['Estimated_Annual_Income']
    X_fe['Claims_Per_Vehicle'] = X_fe['Previous_Claims_Filed'] / (X_fe['Vehicles_on_Policy'] + 1)
    X_fe['Is_Zero_Deductible'] = (X_fe['Deductible_Tier'] == 'Tier_4_Zero_Ded').astype(int)
    X_fe['Is_Full_Time'] = (X_fe['Employment_Status'] == 'Employed_FullTime').astype(int)
    X_fe['Policy_Duration_Bucket'] = pd.cut(
        X_fe['Previous_Policy_Duration_Months'].fillna(0),
        bins=[0, 6, 12, 24, 999],
        labels=[0, 1, 2, 3]
    ).astype(float)

    return X_fe

X = apply_feature_engineering(X)

# Split 80/20
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

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

# Phase VIII Champion hyperparameters
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LGBMClassifier(
        random_state=42,
        class_weight='balanced',
        n_estimators=313,
        learning_rate=0.0770497480076015,
        num_leaves=81,
        max_depth=11,
        min_child_samples=39,
        subsample=0.8613565883831639,
        colsample_bytree=0.7904018562042453,
        reg_alpha=0.08947688579162767,
        reg_lambda=0.01513089539061634,
        verbose=-1
    ))
])

print("🧠 Entraînement du modèle de test en cours...")
model.fit(X_train, y_train)

print("🎯 Génération des prédictions sur les 20% de données cachées...")
y_pred = model.predict(X_val)

# --- RAPPORT TEXTUEL ---
print("\n================ RÉSULTATS DE PRÉCISION ================")
print(classification_report(y_val, y_pred))

# --- VISUALISATION ---
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_val, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)

plt.title('Phase VIII Champion - Matrice de Confusion (63.82% F1)', fontsize=14, pad=20)
plt.xlabel('Prédiction de notre IA', fontsize=12)
plt.ylabel('Réalité (Ce que le client a vraiment acheté)', fontsize=12)
plt.tight_layout()
plt.savefig('accuracy_graph.png', dpi=300)
print("\n✅ Succès ! L'image 'accuracy_graph.png' a été sauvegardée.")
plt.show()
