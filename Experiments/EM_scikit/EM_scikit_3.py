import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from bisect import bisect_left

cols = ['fn_score', 'ln_score', 'age_distance']
# Junget 1850-1845
# data = pd.read_csv(
#     "C:/thesis_code/Github/Experiments/data/junget_1850_1845", usecols=['age_distance', 'fn_score', 'ln_score'])


# Thy 1850-1845
# data = pd.read_csv(
#     "C:/thesis_code/Github/Experiments/data/thy_parishes_1850_1845", usecols=['fn_score', 'ln_score', 'age_distance'])

# Thy 1860-1850
data = pd.read_csv(
    "C:/thesis_code/Github/Experiments/data/thy_parishes_1860_1850", usecols=['fn_score', 'ln_score', 'age_distance'])


class Scikit_EM:
    def __init__(
        self,
        data,
        clusters,
        path,
    ):
        self.data = data
        self.clusters = clusters
        self.path = path
        self.data['fn_score'] = [self.convert_JW(
            feature) for feature in self.data['fn_score']]
        self.data['ln_score'] = [self.convert_JW(
            feature) for feature in self.data['ln_score']]
        self.GaussianMixture()

    def convert_JW(self, feature, breakpoints=[0.75, 0.88, 0.933], values=[3, 2, 1, 0]):
        i = bisect_left(breakpoints, feature)
        return values[i]

    def GaussianMixture(self):
        model = GaussianMixture(
            n_components=self.clusters, init_params='random', random_state=1)
        model.fit(data)
        results = model.predict(self.data)
        self.data['Match'] = results.tolist()
        self.data.reset_index()
        self.data.to_csv(
            f"C:/thesis_code/Github/Experiments/results_after_EM/EM_scikit_3/{self.path}_{self.clusters}", index=False)


#Scikit_EM(data, 2, "thy_parishes_1860_1850")
Scikit_EM(data, 3, "thy_parishes_1860_1850")
