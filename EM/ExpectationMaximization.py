from numpy.lib.npyio import savetxt
import pandas as pd
import numpy as np
from bisect import bisect_left


# dataset = pd.read_csv("C:/thesis_code/Github/data//comp_sets/junget_1850_1845")
# dataset = pd.read_csv("C:/thesis_code/Github/data//comp_sets/testset")
dataset = pd.read_csv("C:/thesis_code/Github/data//comp_sets/testset_2d")

# Bisect dataset before
class ExpectationMaximization:
    def __init__(
        self,
        dataset_array,
        match_guess,
        dimensions,
        elements_dimensions,
        independence=False,
    ):
        self.np_array = dataset_array
        self.dimensions = dimensions
        self.elements_product = np.prod(elements_dimensions)
        self.theta_M = np.full(elements_dimensions, 1 / self.elements_product)
        self.theta_U = np.full(elements_dimensions, 1 / self.elements_product)
        self.p_M = match_guess / len(dataset_array)
        self.p_U = 1 - self.p_M

    def em_steps(self, iterations=100):
        for i in range(iterations):
            w_vector = np.zeros(len(self.np_array))
            # E-STEP
            for i in range(len(self.np_array)):
                theta_value_M = self.theta_M[tuple(self.np_array[i])]
                theta_value_U = self.theta_U[tuple(self.np_array[i])]
                w = (theta_value_M * self.p_M) / (
                    (theta_value_M * self.p_M) + (theta_value_U * self.p_U)
                )
                w_vector[i] = w

            # M-STEP
            for i in range(len(dataset_values)):
                self.theta_M[tuple(self.np_array[i])] = (
                    self.theta_M[tuple(self.np_array[i])] + w_vector[i]
                )
                self.theta_U[tuple(self.np_array[i])] = self.theta_U[
                    tuple(self.np_array[i])
                ] + (1 - w_vector[i])

            # Normalize theta_M and theta_U
            self.theta_M = self.theta_M / self.theta_M.sum()
            self.theta_U = self.theta_U / self.theta_U.sum()

            self.p_M = np.mean(w_vector)
            self.p_U = 1 - self.p_M
        print(self.p_M)
        return self.theta_M


fn_feature = dataset["fn_score"]
ln_feature = dataset["ln_score"]
# dist_age = dataset["age_distance"].astype(int)

# The string numbers below are based on research by Winkler 1988
def convert_JW(feature, breakpoints=[0.75, 0.88, 0.933], values=[3, 2, 1, 0]):
    i = bisect_left(breakpoints, feature)
    return values[i]


dist_fn = [convert_JW(feature) for feature in fn_feature]
dist_ln = [convert_JW(feature) for feature in ln_feature]

# Create numpy array with values for use in E-step
# dataset_values = np.array([dist_age, dist_fn, dist_ln]).transpose()
dataset_values = np.array([dist_fn, dist_ln]).transpose()


# TEST CLASS
print(f"starting CLASS")
em = ExpectationMaximization(dataset_values, 8, 2, [4, 4])
result = em.em_steps()
print(result)
