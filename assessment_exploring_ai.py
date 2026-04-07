import numpy as np
from sklearn.inspection import permutation_importance

# Calculate Importance
perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
sorted_idx = perm_importance.importances_mean.argsort()[-15:]

# Plot
plt.figure(figsize=(10, 8))
plt.barh(X.columns[sorted_idx], perm_importance.importances_mean[sorted_idx])
plt.xlabel("Permutation Importance Score")
plt.title("Top 15 Features Predicting Job Resilience")
plt.tight_layout()
plt.show()