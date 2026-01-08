import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from scipy import sparse

# ============================================================
# Configuration
# ============================================================
INPUT_CSV = '1_student_performance.csv'
PREPROCESSED_CSV = 'student_preprocessed.csv'
PREPROCESSOR_JOBLIB = 'preprocessor.joblib'

GRADE_COLS = ['G1', 'G2', 'G3']
CLASS_TARGET = 'pass'     # classification
REG_TARGET = 'G3'         # regression

# ============================================================
# Load dataset
# ============================================================
if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(f"{INPUT_CSV} not found in {os.getcwd()}")

df = pd.read_csv(
    INPUT_CSV,
    encoding='utf-8',
    low_memory=False,
    on_bad_lines='skip'
)

print('Original shape:', df.shape)

# ============================================================
# Basic validation
# ============================================================
missing = set(GRADE_COLS) - set(df.columns)
if missing:
    raise ValueError(f"Missing required grade columns: {missing}")

# Drop rows without final grade
df = df.dropna(subset=[REG_TARGET])

# ============================================================
# Create targets (DO NOT DROP RAW GRADE)
# ============================================================
# Classification target
df[CLASS_TARGET] = (df[REG_TARGET] >= 10).astype(int)

# ============================================================
# Feature / target separation
# ============================================================
# Exclude targets from features
X = df.drop(columns=[CLASS_TARGET, REG_TARGET])
y_reg = df[REG_TARGET]
y_clf = df[CLASS_TARGET]

# ============================================================
# Column types
# ============================================================
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(exclude=['object']).columns.tolist()

print('Categorical columns:', cat_cols)
print('Numeric columns:', num_cols)

# ============================================================
# Preprocessor (task-agnostic)
# ============================================================
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
        ('num', StandardScaler(), num_cols)
    ],
    remainder='drop'
)

# Fit ONLY on features
preprocessor.fit(X)

# ============================================================
# Transform features
# ============================================================
X_transformed = preprocessor.transform(X)

X_transformed = np.asarray(preprocessor.transform(X))


# Feature names
cat_names = []
try:
    cat_names = (
        preprocessor.named_transformers_['cat']
        .get_feature_names_out(cat_cols)
        .tolist()
    )
except Exception:
    pass

feature_names = cat_names + num_cols

X_preproc_df = pd.DataFrame(
    X_transformed,
    columns=feature_names,
    index=df.index
)

# ============================================================
# Combine features + targets
# ============================================================
preprocessed_df = pd.concat(
    [
        X_preproc_df,
        y_reg.reset_index(drop=True),
        y_clf.reset_index(drop=True)
    ],
    axis=1
)

# ============================================================
# Save artifacts
# ============================================================
preprocessed_df.to_csv(PREPROCESSED_CSV, index=False)
joblib.dump(preprocessor, PREPROCESSOR_JOBLIB)

print('\nSaved files:')
print('-', PREPROCESSED_CSV)
print('-', PREPROCESSOR_JOBLIB)
print('Final dataset shape:', preprocessed_df.shape)
