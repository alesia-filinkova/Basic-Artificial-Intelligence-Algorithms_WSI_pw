from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from logistic_regression import LogisticRegression
from sklearn.preprocessing import StandardScaler

breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)

X = breast_cancer_wisconsin_diagnostic.data.features
# print(X.info())
y = breast_cancer_wisconsin_diagnostic.data.targets

pd.set_option("future.no_silent_downcasting", True)
y = y.replace({"M": 1, "B": 0}).astype(int)

X = X.drop(["radius1", "radius2", "radius3"], axis=1)

X = X.to_numpy()
y = y.to_numpy()

# normalization of data
scaler = StandardScaler()
X = scaler.fit_transform(X)
# X = (X - X.mean(axis=0)) / X.std(axis=0)



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

y_pred_proba = model.predict_proba(X_test)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Accuracy: {accuracy:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"AUROC: {roc_auc:.2f}")
