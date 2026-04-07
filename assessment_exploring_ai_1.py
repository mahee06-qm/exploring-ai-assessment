import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# --- 1. LOAD AND PREPROCESS O*NET RAW DATA ---
df_abilities = pd.read_csv('AB.csv')
df_activities = pd.read_csv('WA.csv')
df_target_source = pd.read_csv('labeled_automation_data.csv')

# Filtering for 'Importance' (IM) to ensure unique feature values per occupation
df_ab_filtered = df_abilities[df_abilities['Scale ID'] == 'IM']
df_wa_filtered = df_activities[df_activities['Scale ID'] == 'IM']

# Pivoting long-format O*NET data into 93 feature columns
features_AB = df_ab_filtered.pivot(index='O*NET-SOC Code', columns='Element Name', values='Data Value')
features_WA = df_wa_filtered.pivot(index='O*NET-SOC Code', columns='Element Name', values='Data Value')

# Merging into a single feature matrix (X)
X_features = pd.merge(features_AB, features_WA, on='O*NET-SOC Code', how='inner')

# --- 2. SYNCHRONIZE FEATURES WITH TARGET LABELS ---
# Aligning features with the 'Probability of automation' from the labeled dataset
df_final = pd.merge(X_features, df_target_source[['O*NET-SOC Code', 'Probability of automation']], on='O*NET-SOC Code', how='inner')

# Defining Target (y): 1 for Resilient (< 0.5 probability), 0 for Augmentable (>= 0.5)
y = df_final['Probability of automation'].apply(lambda x: 1 if x < 0.5 else 0)
X = df_final.drop(columns=['O*NET-SOC Code', 'Probability of automation'])

# --- 3. ADVANCED MACHINE LEARNING PIPELINE ---
# Integrating scaling and classification to prevent data leakage during Cross-Validation
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', probability=True, random_state=42))
])

# Defining hyperparameter grid for RBF Kernel optimization
param_grid = {
    'svm__C': [0.1, 1, 10, 100],
    'svm__gamma': [0.001, 0.01, 0.1, 1]
}

# Implementing Stratified 5-Fold Cross-Validation for robust performance measurement
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=pipeline, 
    param_grid=param_grid, 
    cv=cv_strategy, 
    scoring='f1', 
    n_jobs=-1
)

# --- 4. TRAIN AND EVALUATE ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

grid_search.fit(X_train, y_train)

# --- 5. RESULTS ---
print(f"Mathematically Optimal Parameters: {grid_search.best_params_}")
print("\nAdvanced SVM Model Performance:")
print(classification_report(y_test, grid_search.predict(X_test), target_names=['Augmentable', 'Resilient']))