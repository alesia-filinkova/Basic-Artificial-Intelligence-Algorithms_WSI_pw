from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)

X = breast_cancer_wisconsin_diagnostic.data.features
y = breast_cancer_wisconsin_diagnostic.data.targets

pd.set_option("future.no_silent_downcasting", True)
y = y.replace({"M": 1, "B": 0}).astype(int)


X = X.to_numpy()
y = y.to_numpy()

# normalization of data
scaler = StandardScaler()
X = scaler.fit_transform(X)
# X = (X - X.mean(axis=0)) / X.std(axis=0)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

