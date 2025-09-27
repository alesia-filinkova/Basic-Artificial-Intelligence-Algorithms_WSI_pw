import bnlearn as bn
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

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

def plot_bn_structure(model):
    adjmat = model["adjmat"]
    G = nx.from_pandas_adjacency(adjmat, create_using=nx.DiGraph)
    plt.figure(figsize=(10, 7))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=3000, 
            node_color="lightblue", edge_color="gray", 
            font_size=10, font_weight="bold", arrows=True)
    plt.title("Bayesian Network Structure", fontsize=14)
    plt.show()
    # print(adjmat)

# plot_bn_structure(model)

def generate_observation(model, partial_obs):    
    sample = partial_obs.copy()
    known_data = {k: v for k, v in partial_obs.items() if v != "?"}
    
    for variable in sample:  
        if sample[variable] == "?":
            # get probability of possible values
            prob_dist = bn.inference.fit(model, variables=[variable], evidence=known_data)
            prob_dist_dict = dict(zip(prob_dist.state_names[variable], prob_dist.values))
            if prob_dist_dict and variable in prob_dist_dict:
                most_probable_value = max(prob_dist[variable], key=prob_dist[variable].get)
                sample[variable] = most_probable_value
            else:
                sample[variable] = np.random.choice(list(model['model'].states[variable]))

    return sample


partial_observation = {"Victim Sex": "?",
                        "Victim Age": "20",
                        "Victim Race": "?", 
                        "Perpetrator Sex": "Male", 
                        "Perpetrator Age": "?", 
                        "Perpetrator Race": "Asian/Pacific Islander", 
                        "Relationship": "Friend", 
                        "Weapon": "Strangulation"}
print(generate_observation(model, partial_observation))
