import bnlearn as bn
import pandas as pd

url = "US_Crime_DataSet.csv"
data = pd.read_csv(url)

target_columns = [
    "Victim Sex", "Victim Age", "Victim Race",
    "Perpetrator Sex", "Perpetrator Age", "Perpetrator Race",
    "Relationship", "Weapon"
]
data = data[target_columns]

data = data[~data.isin(["Unknown"]).any(axis=1)]

