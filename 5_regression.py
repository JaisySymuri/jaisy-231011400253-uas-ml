import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# Configuration
# ============================================================
PREPROCESSED_CSV = 'student_preprocessed.csv'
MODEL_OUT = 'decision_tree_regressor.joblib'

TARGET_COL = 'G3'
LEAKAGE_COLS = ['pass']  # classification target must not be used

# ============================================================
# Load dataset
# ============================================================
if not os.path.exists(PREPROCESSED_CSV):
    raise FileNotFoundError('Run preprocessing script first.')

df = pd.read_csv(PREPROCESSED_CSV)
print('Dataset shape:', df.shape)

if TARGET_COL not in df.columns:
    raise ValueError("Target column 'G3' not found")

# ============================================================
# Feature / target split
# ============================================================
X = df.drop(columns=[TARGET_COL] + LEAKAGE_COLS)
y = df[TARGET_COL]

print('Feature shape:', X.shape)
print('Target stats:')
print(y.describe())

# ============================================================
# Train / test split
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ============================================================
# Hyperparameter tuning — Decision Tree Regressor
# ============================================================
param_grid = {
    'criterion': ['squared_error', 'absolute_error'],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 5, 10]
}

grid_search = GridSearchCV(
    estimator=DecisionTreeRegressor(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_tree = grid_search.best_estimator_

print("\nBest Decision Tree Regressor Parameters:")
print(grid_search.best_params_)

# ============================================================
# Model evaluation
# ============================================================
y_pred = best_tree.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\n=== Decision Tree Regression Results ===")
print("MAE :", mae)
print("RMSE:", rmse)
print("R²  :", r2)

# ============================================================
# Save model
# ============================================================
joblib.dump(best_tree, MODEL_OUT)
print("\nSaved model:", MODEL_OUT)
