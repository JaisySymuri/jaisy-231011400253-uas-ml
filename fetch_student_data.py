from ucimlrepo import fetch_ucirepo
import pandas as pd

student_performance = fetch_ucirepo(id=320)

if student_performance is not None and student_performance.data is not None:
    X = student_performance.data.features
    y = student_performance.data.targets
    df = pd.concat([X, y], axis=1)
    df.to_csv('student_performance.csv', index=False)
    print("✅ Dataset fetched and saved as student_performance.csv")
else:
    print("❌ Failed to fetch dataset (id=320)")
