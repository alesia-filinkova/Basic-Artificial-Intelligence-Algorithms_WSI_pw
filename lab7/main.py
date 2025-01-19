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

data["Victim Age"] = data["Victim Age"].astype(str)
data["Perpetrator Age"] = data["Victim Age"].astype(str)

data = data[~data.isin(["Unknown"]).any(axis=1)]

# Uczenie struktury sieci
model = bn.structure_learning.fit(data, methodtype='hc')

# Uczenie parametrów sieci
model = bn.parameter_learning.fit(model, data)

# Funkcja generowania danych na podstawie niepełnej obserwacji
def generate_observation(model, partial_obs):
    sample = bn.sampling(model, n=1).iloc[0].to_dict()
    for key, value in partial_obs.items():
        if value != "?":
            sample[key] = value
    return sample


partial_observation = {"Victim Sex": "?",
                        "Victim Age": "20",
                        "Victim Race": "?", 
                        "Perpetrator Sex": "male", 
                        "Perpetrator Age": "?", 
                        "Perpetrator Race": "asian", 
                        "Relationship": "friend", 
                        "Weapon": "strangulation"}
print(generate_observation(model, partial_observation))
