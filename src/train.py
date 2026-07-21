import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,confusion_matrix)
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from imblearn.ensemble import BalancedRandomForestClassifier
df = pd.read_csv("data/processed/cicids2017_processed.csv")

X = df.drop("Label", axis=1)
y = df["Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = BalancedRandomForestClassifier(
    n_estimators=500,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

probs = model.predict_proba(X_test)[:, 1]

threshold = 0.5

y_pred = (probs >= threshold).astype(int)


print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

importance = pd.Series(
    model.feature_importances_,
    index=X_train.columns
)

print(importance.sort_values(ascending=False))
joblib.dump(model, "models/random_forest_hikari.pkl")

