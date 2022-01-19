from numpy.lib.npyio import savetxt
import pandas as pd
import numpy as np
from bisect import bisect_left


dataset = pd.read_csv("C:/thesis_code/Github/data//comp_sets/junget_1850_1845")
#dataset = pd.read_csv("C:/thesis_code/Github/data//comp_sets/testset")
#dataset = pd.read_csv("C:/thesis_code/Github/data//comp_sets/testset_2d")


class ExpectationMaximization:
    def __init__(
        self,
        data,
        match_guess,
        dimensions,
        elements_dimensions,
        independence=False,
    ):
        self.data = data
        self.dimensions = dimensions
        self.elements_product = np.prod(elements_dimensions)
        self.theta_M = np.full(elements_dimensions, 1 / self.elements_product)
        self.theta_U = np.full(elements_dimensions, 1 / self.elements_product)
        self.p_M = match_guess / len(data)
        self.p_U = 1 - self.p_M
        self.geo_dist_M = self.geometric_dist(0.5)
        self.geo_dist_U = self.geo_dist_M[::-1]

    def em_steps(self, iterations=100):
        for i in range(iterations):
            w_vector = np.zeros(len(self.data))
            # E-STEP
            for i in range(len(self.data)):
                theta_value_M = self.theta_M[tuple(self.data[i])]
                theta_value_U = self.theta_U[tuple(self.data[i])]
                w = (theta_value_M * self.weighted_p(tuple(self.data[i]), self.geo_dist_M)) / (
                    (theta_value_M * self.weighted_p(tuple(self.data[i]), self.geo_dist_M)) + (
                        theta_value_U * self.weighted_p(tuple(self.data[i]), self.geo_dist_U))
                )
                # w = (theta_value_M * self.p_M) / (
                #    (theta_value_M * self.p_M) + (theta_value_U * self.p_U)
                # )
                w_vector[i] = w

            # M-STEP
            for i in range(len(dataset_values)):
                self.theta_M[tuple(self.data[i])] = (
                    self.theta_M[tuple(self.data[i])] + w_vector[i]
                )
                self.theta_U[tuple(self.data[i])] = self.theta_U[
                    tuple(self.data[i])
                ] + (1 - w_vector[i])

            # Normalize theta_M and theta_U
            self.theta_M = self.theta_M / self.theta_M.sum()
            self.theta_U = self.theta_U / self.theta_U.sum()

            #self.p_M = np.mean(w_vector)
            #self.p_U = 1 - self.p_M
        print(self.theta_U)
        return self.theta_M

    def geometric_dist(self, p):
        list = [p, (1-p)*p, (1-p)**2*p, (1-p)**3*p]
        return list

    def weighted_p(self, tuple, geo_dist):
        a, b = tuple
        p = geo_dist[a] * geo_dist[b]
        return p

    def bisect(self, dataset, age_included=True):
        if age_included == True:
            pass

    def sanity_check(self):
        pass  # implement the sanity check from the paper


fn_feature = dataset["fn_score"]
ln_feature = dataset["ln_score"]
#dist_age = dataset["age_distance"].astype(int)


def convert_JW(feature, breakpoints=[0.75, 0.88, 0.933], values=[3, 2, 1, 0]):
    i = bisect_left(breakpoints, feature)
    return values[i]


dist_fn = [convert_JW(feature) for feature in fn_feature]
dist_ln = [convert_JW(feature) for feature in ln_feature]

# dataset_values = np.array([dist_age, dist_fn, dist_ln]).transpose()
dataset_values = np.array([dist_fn, dist_ln]).transpose()


# TEST CLASS
print(f"starting CLASS")
em = ExpectationMaximization(dataset_values, 100, 2, [4, 4])
result = em.em_steps(1000)
print(result)
