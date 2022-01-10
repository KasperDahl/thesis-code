from numpy.lib.npyio import savetxt
import pandas as pd
import numpy as np
from bisect import bisect_left

dataset = pd.read_csv("C:/thesis_code/Github/data//comp_sets/junget_1850_1845")
# dataset = pd.read_csv("C:/thesis_code/Github/data//comp_sets/testset")

# Bisect dataset before
class ExpectationMaximization:
    def __init__(
        self,
        np_array,
        match_guess,
        dimensions,
        elements_dimensions,
        independence=False,
    ):
        self.np_array = np_array
        self.dimensions = dimensions
        self.elements_product = np.prod(elements_dimensions)
        # Spørgsmål Nicolai: Hvorfor skal jeg bruge self-argumentet inde i parentesen nedenunder?
        self.theta_M = np.full(elements_dimensions, 1 / self.elements_product)
        self.theta_U = np.full(elements_dimensions, 1 / self.elements_product)
        self.p_M = match_guess / len(np_array)
        # I forlængelse af ovenstående spørgsmål: Jeg behøver tilsyneladende ikke self for p_M nedenunder - hvorfor?
        self.p_U = 1 - self.p_M
        print(self.p_M)

    def em_steps(self, iterations=100):
        # E-STEP
        print(len(self.np_array))
        for i in range(iterations):
            w_vector = np.zeros(len(self.np_array))
            for i in range(len(self.np_array)):
                theta_value_M = self.theta_M[tuple(self.np_array[i])]
                theta_value_U = self.theta_U[tuple(self.np_array[i])]
                # vector w
                w = (theta_value_M * self.p_M) / (
                    (theta_value_M * self.p_M) + (theta_value_U * self.p_U)
                )
                w_vector[i] = w

        # M-STEP
        # print("start M-step")
        # maximize theta values in 3-d arrays
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
        return self.theta_M


fn_feature = dataset["fn_score"]
ln_feature = dataset["ln_score"]
# not entirely sure why this list is received as floats
dist_age = dataset["age_distance"].astype(int)
# 1. Distance Bins

# The string numbers below are based on research by Winkler 1988


def convert_JW(feature, breakpoints=[0.75, 0.88, 0.933], values=[3, 2, 1, 0]):
    i = bisect_left(breakpoints, feature)
    return values[i]


dist_fn = [convert_JW(feature) for feature in fn_feature]
dist_ln = [convert_JW(feature) for feature in ln_feature]


# Create numpy array with values for use in E-step
dataset_values = np.array([dist_age, dist_fn, dist_ln]).transpose()


# TEST CLASS
print(f"starting CLASS")
em = ExpectationMaximization(dataset_values, 100, 3, [3, 4, 4])
result = em.em_steps()
print(result)


# 2. Parameters

# 3-dimensional array consisting of the multinomial distance bins (fn, ln, age)
# LOOK INTO: how to automate this (could be min-max range calculation, or convert to sets - otherwise just received as parameters)
dimensions = 3 * 4 * 4
theta_M = np.full((3, 4, 4), 1 / dimensions)
theta_U = np.full((3, 4, 4), 1 / dimensions)


# probability of being a match - should be a reasonable guess
# example: if I estimate a 100 matches in the comparison space of two censuses of a 1000 records each
# the estimated probability is 100/(1000 X 1000) = 0,0001
match_guess = 100
# match_guess = 8
p_M = match_guess / len(dataset)
p_U = 1 - p_M
# print(p_M)
# 3. Loop over steps E and M


def em_steps(p_M, p_U, theta_M, theta_U):

    # E-STEP
    # initialize loop - number of iterations will be changed later
    # loop until convergence - or maximum 100 iterations (or other number)
    # if the latter - this information should be outputted
    n_iterations = 100
    print(len(dataset_values))
    for i in range(n_iterations):
        w_vector = np.zeros(len(dataset_values))

        # print("start E-step")
        # get features and look up corresponding theta element for pair
        for i in range(len(dataset_values)):
            distances = dataset_values[i]

            theta_value_M = theta_M[distances[0], distances[1], distances[2]]
            theta_value_U = theta_U[distances[0], distances[1], distances[2]]
            # vector w
            w = (theta_value_M * p_M) / ((theta_value_M * p_M) + (theta_value_U * p_U))
            w_vector[i] = w

        # print("end E-step")

        # M-STEP
        # print("start M-step")
        # maximize theta values in 3-d arrays
        for i in range(len(dataset_values)):
            distances = dataset_values[i]
            theta_M[distances[0], distances[1], distances[2]] = (
                theta_M[distances[0], distances[1], distances[2]] + w_vector[i]
            )
            theta_U[distances[0], distances[1], distances[2]] = theta_U[
                distances[0], distances[1], distances[2]
            ] + (1 - w_vector[i])

        # Normalize theta_M and theta_U
        theta_M = theta_M / theta_M.sum()
        theta_U = theta_U / theta_U.sum()

        p_M = np.mean(w_vector)
        p_U = 1 - p_M

        # print(f"P_M: {p_M}, P_U: {p_U}")
    # print(f"Theta_U: \n {theta_U}")
    print(p_M)
    return theta_M


print(f"starting NOT CLASS")
theta_final = em_steps(p_M, p_U, theta_M, theta_U)
print(theta_final)

np.save("C:/thesis_code/Github/data/numpy_arrays/np_array1.npy", theta_final)
