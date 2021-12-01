# prepping small trainingsets for testing of EM-algorithm
# Initially the following parishes from 1845 and 1850 censuses:
# Sevel(1845: 1155, 1850: 1162), Selde(1845: 471, 1850: 491), Thorum(1845: 462, 1850: 500), Junget(1845: 264, 1850: 281)

import pandas as pd

c1845 = pd.read_csv("C:/thesis_code/Github/data/censuses/census_1845")
c1850 = pd.read_csv("C:/thesis_code/Github/data/censuses/census_1850")

parishes = ["sevel", "selde", "thorum", "junget"]

# create new df of people from specific parishes and add new column with birth year
for p in parishes:
    df = c1845.loc[c1845["event_parish"] == p]
    print(f"{p}, {df.shape}")
    df['birth_year'] = 1845 - df['age']
    df.reset_index()
    df.to_csv(f"C:/thesis_code/Github/data/trainingsets_s/{p}_1845")


for p in parishes:
    df = c1850.loc[c1850["event_parish"] == p]
    print(f"{p}, {df.shape}")
    df['birth_year'] = 1850 - df['age']
    df.reset_index()
    df.to_csv(f"C:/thesis_code/Github/data/trainingsets_s/{p}_1850")
