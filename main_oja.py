import numpy as np
import pandas as pd
import json

europe_df = pd.read_csv('data/europe.csv')
numerical_cols = europe_df.select_dtypes(include=np.number).columns
df = europe_df[numerical_cols]
numerical_inputs = df.values.astype(np.float64)
labels = europe_df["Country"].tolist()

with open('configs/oja_config.json', 'r') as f:
    config = json.load(f)
from src.OjaNet import OjaNet

oja_net = OjaNet(
    numerical_inputs,
    config['max_epochs'],
    config['ini_learn_rate'],
    config['decrease_learn_rate']
)

while oja_net.has_next():
    oja_net.next_epoch()

print("The converged weights were:")
print(oja_net.weights)