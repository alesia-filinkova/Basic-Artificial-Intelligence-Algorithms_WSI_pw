from ucimlrepo import fetch_ucirepo
import pandas as pd

breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)

X = breast_cancer_wisconsin_diagnostic.data.features
y = breast_cancer_wisconsin_diagnostic.data.targets

pd.set_option("future.no_silent_downcasting", True)
y = y.replace({"M": 1, "B": 0}).astype(int)


X = X.to_numpy()
y = y.to_numpy()
