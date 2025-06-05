import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from src.KohonenPlotUtils import plot_activation_map, plot_average_euclidean_distance_of_weights, plot_average_variable
from src.Kohonen import KohonenNet

europe_df = pd.read_csv('data/europe.csv')
numerical_cols = europe_df.select_dtypes(include=np.number).columns
df = europe_df[numerical_cols]
numerical_inputs = df.values.astype(np.float64)
labels = europe_df["Country"].tolist()

with open('configs/kohonen_config.json', 'r') as f:
    config = json.load(f)

kn = KohonenNet(
    numerical_inputs, 
    config['k_grid_dimension'],
    config['use_weights_from_inputs'],
    config['max_epochs'],
    config['initial_radius'],
    config['decrease_radius'],
    config['initial_learn_rate'],
    config['decrease_learn_rate'],
    labels=labels
)

while kn.has_next():
    kn.next_epoch()

plot_type = config['plot_type'].lower()
if plot_type == 'final_entries':
    plot_activation_map(kn.activations)
elif plot_type == 'euclidean':
    plot_average_euclidean_distance_of_weights(kn)
elif plot_type == 'variable':
    if config['variable_type'] is None:
        raise Exception('Missing variable category')
    
    parse = {
        "military": "Military",
        "unemployment": "Unemployment",
        "life.expect": "Life.expect",
        "area": "Area",
        "gdp": "GDP",
        "inflation": "Inflation",
        "pop.growth": "Pop.growth"
    }
    
    plot_average_variable(
        kn.activations,
        parse[config['variable_type'].lower()],
        kn,
        labels,
        europe_df
    )
else:
    raise Exception('Missing or wrong plot type')