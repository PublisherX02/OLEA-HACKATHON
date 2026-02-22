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

print("🤖 Démarrage de l'Intelligence Artificielle Optuna...")
print("📊 Chargement des données d'entraînement...")
df = pd.read_csv("train.csv")
X = df.drop(columns=['Purchased_Coverage_Bundle', 'User_ID'])
y = df['Purchased_Coverage_Bundle']

# --- FEATURE ENGINEERING (Identique à notre solution championne) ---
def apply_feature_engineering(df):
    X_fe = df.copy()
    X_fe['Total_Dependents'] = X_fe['Adult_Dependents'] + X_fe['Child_Dependents'].fillna(0) + X_fe['Infant_Dependents']
    X_fe['Income_per_Dependent'] = X_fe['Estimated_Annual_Income'] / (X_fe['Total_Dependents'] + 1)
    X_fe['Risk_Ratio'] = X_fe['Previous_Claims_Filed'] / (X_fe['Years_Without_Claims'] + 1)
    X_fe['Vehicles_per_Adult'] = X_fe['Vehicles_on_Policy'] / (X_fe['Adult_Dependents'] + 1)
    return X_fe

X = apply_feature_engineering(X)

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

def objective(trial):
    """
    La fonction 'objective' qu'Optuna va chercher à MAXIMISER (Macro F1 Score)
    """
    # 1. Optuna choisit de nouveaux hyperparamètres à tester pour cette itération
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 5, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'random_state': 42,
        'class_weight': 'balanced',
        'verbose': -1
    }
    
    # 2. On évalue le modèle avec ces paramètres secrets en utilisant la Cross-Validation
    # k=3 est assez robuste pour juger si ces paramètres sont bons
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    f1_scores = []
    
    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LGBMClassifier(**params))
        ])
        
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        
        # Le juge utilise le F1 Macro !
        score = f1_score(y_val, preds, average='macro')
        f1_scores.append(score)
        
    return np.mean(f1_scores)

# On demande à Optuna de créer une étude pour MAXIMISER le score
study = optuna.create_study(direction="maximize", study_name="LightGBM_Insurance")

# Callback pour arrêter la recherche si on dépasse 63% de Macro F1 (Score exceptionnel !)
def stop_early(study, trial):
    if study.best_value >= 0.63:
        print(f"\n🛑 OBJECTIF ATTEINT : {study.best_value:.4f} > 0.63. Arrêt immédiat de la recherche pour gagner du temps !")
        study.stop()

# On limite la recherche à 50 combinaisons (ça prendra env 3-5 minutes)
print("🚀 Lancement brute-force sur 50 combinaisons d'hyperparamètres...")
study.optimize(objective, n_trials=50, callbacks=[stop_early])

print("\n🏆 OPTUNA TERMINE 🏆")
print(f"🥇 Meilleur Macro F1 Obtenu : {study.best_value:.4f}")
print("⚙️ Paramètres gagnants à injecter dans build_model.py :")
for key, value in study.best_params.items():
    print(f"    {key}={value},")
