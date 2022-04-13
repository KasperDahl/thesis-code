import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from bisect import bisect_left


# Manual 1850-1845
# data = pd.read_csv(
#     "C:/thesis_code/Github/Experiments/data/manual_1850_1845", usecols=['age_distance', 'fn_score', 'ln_score', 'fam_n_score', 'bp_score'])

# Manual 1860-1850
data = pd.read_csv(
    "C:/thesis_code/Github/Experiments/data/manual_1860_1850", usecols=['age_distance', 'fn_score', 'ln_score', 'fam_n_score', 'bp_score'])


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
        self.data['fam_n_score'] = [self.convert_JW(
            feature) for feature in self.data['fam_n_score']]
        self.data['bp_score'] = [self.convert_JW(
            feature) for feature in self.data['bp_score']]
        self.GaussianMixture()

    def convert_JW(self, feature, breakpoints=[0.75, 0.88, 0.933], values=[3, 2, 1, 0]):
        i = bisect_left(breakpoints, feature)
        return values[i]

    def GaussianMixture(self):
        model = GaussianMixture(
            #    n_components=self.clusters, init_params='random', random_state=1)
            # random state needs to be removed for 3 cluster version - otherwise it reverses feature values
            n_components=self.clusters, init_params='random')
        model.fit(data)
        results = model.predict(self.data)
        self.data['Match'] = results.tolist()
        self.data.reset_index()
        self.data.to_csv(
            f"C:/thesis_code/Github/Experiments/results_after_EM/EM_scikit_5/{self.path}_{self.clusters}", index=False)


# Scikit_EM(data, 2, "manual_1850_1845")
# Scikit_EM(data, 3, "manual_1850_1845")

# Scikit_EM(data, 2, "manual_1860_1850")
Scikit_EM(data, 3, "manual_1860_1850")
