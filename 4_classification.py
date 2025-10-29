import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
PREPROCESSED_CSV = 'student_preprocessed.csv'
PREPROCESSOR_JOBLIB = 'preprocessor.joblib'
TREE_MODEL = 'decision_tree.joblib'
SVM_MODEL = 'svm_model.joblib'

# ---------------------------------------------------------------------
# Safety check
# ---------------------------------------------------------------------
if not os.path.exists(PREPROCESSED_CSV) or not os.path.exists(PREPROCESSOR_JOBLIB):
    raise FileNotFoundError('Preprocessed files not found. Run the preprocessing script first.')

# ---------------------------------------------------------------------
# Load preprocessed data
# ---------------------------------------------------------------------
df = pd.read_csv(PREPROCESSED_CSV)
print('Dataset shape:', df.shape)

if 'pass' not in df.columns:
    raise ValueError("Expected 'pass' column in preprocessed CSV")

X = df.drop(columns=['pass'])
y = df['pass']

# ---------------------------------------------------------------------
# Train/test split
# ---------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------------------
# Load fitted preprocessor (for reference only)
# ---------------------------------------------------------------------
preprocessor = joblib.load(PREPROCESSOR_JOBLIB)

# ---------------------------------------------------------------------
# Decision Tree Classifier
# ---------------------------------------------------------------------
tree = DecisionTreeClassifier(max_depth=5, random_state=42)
tree.fit(X_train, y_train)

y_pred_tree = tree.predict(X_test)
print("\n=== Decision Tree Results ===")
print("Accuracy:", accuracy_score(y_test, y_pred_tree))
print(classification_report(y_test, y_pred_tree))

# ---------------------------------------------------------------------
# Support Vector Machine (SVM) Classifier
# ---------------------------------------------------------------------
svm = SVC(kernel='rbf', C=1.0, gamma='scale')
svm.fit(X_train, y_train)

y_pred_svm = svm.predict(X_test)
print("\n=== SVM Results ===")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))

# ---------------------------------------------------------------------
# Save models
# ---------------------------------------------------------------------
joblib.dump(tree, TREE_MODEL)
joblib.dump(svm, SVM_MODEL)
print('\nSaved models:', TREE_MODEL, 'and', SVM_MODEL)
