import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import numpy as np
from scipy import sparse


INPUT_CSV = '1-student_performance.csv'
PREPROCESSED_CSV = 'student_preprocessed.csv'
PREPROCESSOR_JOBLIB = 'preprocessor.joblib'

if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(f"{INPUT_CSV} not found in working dir: {os.getcwd()}")

df = pd.read_csv(INPUT_CSV, low_memory=False, encoding='utf-8', on_bad_lines='skip')
print('shape:', df.shape)
print(df.head())


required_grade_cols = {'G1','G2','G3'}
if not required_grade_cols.issubset(df.columns):
    print('Warning: expected grade columns not found. Current columns:', df.columns.tolist())

# Drop rows with all-NaN or missing final grade
if 'G3' in df.columns:
    df = df.dropna(subset=['G3'])

# Create binary target: pass if G3 >= 10 (change threshold if you prefer)
df['pass'] = (df['G3'] >= 10).astype(int)

# Optionally drop the raw grades to avoid leakage
if set(['G1','G2','G3']).issubset(df.columns):
    df = df.drop(columns=['G1','G2','G3'])

# 01_preprocessing.ipynb - cell 5
# Separate X and y
X = df.drop(columns=['pass'])
y = df['pass']

# Identify column types
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(exclude=['object']).columns.tolist()

print('categorical columns:', cat_cols)
print('numeric columns:', num_cols)

# 01_preprocessing.ipynb - cell 6
# Build preprocessor. OneHotEncoder for categoricals, StandardScaler for numerics.
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
        ('num', StandardScaler(), num_cols)
    ],
    remainder='drop'
)

preprocessor.fit(X)

try:
    cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols).tolist()
except Exception:
    # For older sklearn versions
    cat_names = []
    if hasattr(preprocessor.named_transformers_['cat'], 'get_feature_names'):
        cat_names = preprocessor.named_transformers_['cat'].get_feature_names(cat_cols).tolist()

feature_names = cat_names + num_cols
X_transformed = preprocessor.transform(X)

if sparse.issparse(X_transformed):
    # Tell Pylance: inside this block, it's definitely sparse
    X_transformed = sparse.csr_matrix(X_transformed).toarray()
else:
    X_transformed = np.asarray(X_transformed)

X_preproc_df = pd.DataFrame(X_transformed, columns=feature_names, index=X.index)

# Combine with target
preprocessed = pd.concat([X_preproc_df, y.reset_index(drop=True)], axis=1)

# 01_preprocessing.ipynb - cell 7
# Save preprocessed CSV and the fitted preprocessor
preprocessed.to_csv(PREPROCESSED_CSV, index=False)
joblib.dump(preprocessor, PREPROCESSOR_JOBLIB)

print('Saved:', PREPROCESSED_CSV, PREPROCESSOR_JOBLIB)


