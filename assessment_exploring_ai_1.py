import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.inspection import permutation_importance

# Load and refine O*NET data by filtering for 'Importance' to ensure unique indices
df_ab = pd.read_csv('AB.csv')
df_wa = pd.read_csv('WA.csv')
df_target = pd.read_csv('labeled_automation_data.csv')

df_ab_im = df_ab[df_ab['Scale ID'] == 'IM']
df_wa_im = df_wa[df_wa['Scale ID'] == 'IM']

# Pivot and merge to create a 93-feature matrix
features_ab = df_ab_im.pivot(index='O*NET-SOC Code', columns='Element Name', values='Data Value')
features_wa = df_wa_im.pivot(index='O*NET-SOC Code', columns='Element Name', values='Data Value')
X_all = pd.merge(features_ab, features_wa, on='O*NET-SOC Code', how='inner')

# Align features with automation labels
df_final = pd.merge(X_all, df_target[['O*NET-SOC Code', 'Probability of automation']], on='O*NET-SOC Code', how='inner')
X = df_final.drop(columns=['O*NET-SOC Code', 'Probability of automation'])
y = df_final['Probability of automation'].apply(lambda x: 1 if x < 0.5 else 0)

# Build a sophisticated Pipeline with RBF SVM and GridSearch optimization
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', probability=True, random_state=42))
])

param_grid = {'svm__C': [0.1, 1, 10, 100], 'svm__gamma': [0.001, 0.01, 0.1, 1]}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='f1', n_jobs=-1)

# Train and execute advanced evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
grid_search.fit(X_train, y_train)

# Output detailed metrics and feature importance
best_model = grid_search.best_estimator_
print(f"Optimal Parameters: {grid_search.best_params_}")
print(f"ROC-AUC Score: {roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1]):.4f}")
print(classification_report(y_test, best_model.predict(X_test)))

# Permutation importance for model interpretability
importance = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=42)
print("Top Predictive Skill for Resilience:", X.columns[np.argmax(importance.importances_mean)])