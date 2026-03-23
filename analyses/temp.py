import pandas as pd 

collected = pd.read_csv('data/combined.csv')

print(collected['AI Knowledge'].unique())