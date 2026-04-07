from sklearn.preprocessing import StandardScaler, LabelEncoder

# 1. Separate Features (X) and Target (y)
# X contains the 93 skill/activity features; y contains the Risk_Level
X = df.loc[:, 'Arm-Hand Steadiness':'Working with Computers']
y = df['Risk_Level']

# 2. Label Encoding (Categorical to Numerical)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 3. Feature Scaling (Standardization for SVM)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Results
print(f"Features (X) shape: {X_scaled.shape}")
print(f"Target (y) classes: {le.classes_}")
print(f"Mapping: {dict(zip(le.classes_, range(len(le.classes_))))}")