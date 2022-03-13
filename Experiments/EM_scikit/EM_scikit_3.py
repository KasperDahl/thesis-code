import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from bisect import bisect_left

cols = ['fn_score', 'ln_score', 'age_distance']
data = pd.read_csv(
    "C:/thesis_code/Github/data/comp_sets/thy_parishes_1850_1845", usecols=cols)
# data = pd.read_csv(
#    "C:/thesis_code/Github/data/comp_sets/junget_1850_1845", usecols=cols)


def convert_JW(feature, breakpoints=[0.75, 0.88, 0.933], values=[3, 2, 1, 0]):
    i = bisect_left(breakpoints, feature)
    return values[i]


data['fn_score'] = [convert_JW(feature) for feature in data['fn_score']]
data['ln_score'] = [convert_JW(feature) for feature in data['ln_score']]

data.to_csv(
    "C:/thesis_code/Github/data/comp_sets/thy_parishes_1850_1845_bisected")

model = GaussianMixture(n_components=3, init_params='random', random_state=1)
model.fit(data)
results = model.predict(data)

df = pd.read_csv("C:/thesis_code/Github/data/comp_sets/thy_parishes_1850_1845",
                 usecols=['Unnamed: 0', 'Unnamed: 1'])
df['EM_results'] = results.tolist()
df.reset_index()
# print(type(df))
print(df)

#np.savetxt("C:/thesis_code/Github/data/results/junget_1850_1845_EM_scikit", results)
df.to_csv(
    "C:/thesis_code/Github/data/results/thy_parishes_1850_1845_EM_scikit", index=False)
# print(yhat[:100])
