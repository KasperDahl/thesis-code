from numpy.lib.npyio import savetxt
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from bisect import bisect_left
from time import perf_counter as pc


# dataset = pd.read_csv(
#    "C:/thesis_code/Github/data/comp_sets/thy_parishes_1850_1845")
dataset = pd.read_csv("C:/thesis_code/Github/data/comp_sets/junget_1850_1845")
# dataset = pd.read_csv("C:/thesis_code/Github/data/comp_sets/testset")
# dataset = pd.read_csv("C:/thesis_code/Github/data/comp_sets/testset_2d")


class ExpectationMaximization:
    def __init__(
        self,
        data,
    ):
        self.data = data
        self.P_A_M = self.P_FN_M = self.P_LN_M = 0.5
        self.P_A_U = self.P_FN_U = self.P_LN_U = 0.5

    def em_steps(self, iterations=100):
        for i in range(iterations):
            w_vector = np.zeros(len(self.data))
            # E-STEP
            for i in range(len(self.data)):
                # The three P values are multiplied with the respective geometric distribution value based on their feature value;
                # these variables are then used to find w for those given feature values
                p_A_M = self.geo_dist_age(
                    self.P_A_M, self.data[i][0])
                p_A_U = self.geo_dist_age(
                    self.P_A_U, self.data[i][0], matches=False)
                p_FN_M = self.geo_dist_names(
                    self.P_FN_M, self.data[i][1])
                p_FN_U = self.geo_dist_names(
                    self.P_FN_U, self.data[i][1], matches=False)
                p_LN_M = self.geo_dist_names(
                    self.P_LN_M, self.data[i][2])
                p_LN_U = self.geo_dist_names(
                    self.P_LN_U, self.data[i][2], matches=False)

                w = (p_A_M*p_FN_M*p_LN_M) / \
                    ((p_A_M*p_FN_M*p_LN_M)+(p_A_U*p_FN_U*p_LN_U))
                w_vector[i] = w

            # remove for loop
            # Run minize instead
            # Loop inside the objective function

            x_M = [self.P_A_M, self.P_FN_M, self.P_LN_M]
            x_U = [self.P_A_U, self.P_FN_U, self.P_LN_U]

            x_M_1 = minimize(self.L_M, x0=x_M, args=(self.data, w_vector),
                             method='L-BFGS-B', bounds=[(0, 1), (0, 1), (0, 1)])
            x_U_1 = minimize(self.L_M, x0=x_U, args=(self.data, w_vector),
                             method='L-BFGS-B', bounds=[(0, 1), (0, 1), (0, 1)])

            self.P_A_M = x_M_1[0]
            self.P_FN_M = x_M_1[1]
            self.P_LN_M = x_M_1[2]

            self.P_A_U = x_U_1[0]
            self.P_FN_U = x_U_1[1]
            self.P_LN_U = x_U_1[2]

        return 1

    def L_M(self, x, data, w_vector):
        log_p_a = np.log(x[0]+x[0]**2+x[0]**3)
        log_p_f = np.log(x[1]+x[1]**2+x[1]**3+x[1]**4)
        log_p_l = np.log(x[2]+x[2]**2+x[2]**3+x[2]**4)
        for i in range(len(data)):
            w_vector[i]*(data[i][0]*np.log(x[0])-log_p_a+data[i][1]
                         * np.log(x[1])-log_p_f+data[i][2]*np.log(x[2])-log_p_l)
        return 1

        # pass the dataset and iterate over the training set and pull out the relevant values
        # Inside the objective function I can compute log_p_f, log_p_l, log_p_a initially in the objective function
        # I can also precompute the

    # def bayes_conversion(self, theta_M):
    #     dist = np.empty(self.elements_dimensions)
    #     length = len(self.data)
    #     for x, v in np.ndenumerate(dist):
    #         dist[x] = (theta_M[x]*self.geo_dist_M[x]) /
    #         ((theta_M[x]*self.geo_dist_M[x]) +
    #          (self.theta_U[x]*self.geo_dist_U[x]))
    #     return dist

    # function to calculate geometric distribution in the E-step
    # p is the given probability, k is the feature value
    def geo_dist_names(self, p, k, matches=True):
        if not matches:
            k = 3 - k
        return (p**(1+k))/(p+p**2+p**3+p**4)

    def geo_dist_age(self, p, k, matches=True):
        if not matches:
            k = 3 - k
        return (p**(1+k))/(p+p**2+p**3)

    # def geo_dist_names(self, p, k, matches=True):
    #     sum = 0
    #     for i in range(1, 5):
    #         sum += p**i
    #     if not matches:
    #         k = 3 - k
    #     return (p**(1+k))/sum

    # def geo_dist_age(self, p, k, matches=True):
    #     sum = 0
    #     for i in range(1, 4):
    #         sum += p**i
    #     if not matches:
    #         k = 2 - k
    #     return (p**(1+k))/sum


fn_feature = dataset["fn_score"]
ln_feature = dataset["ln_score"]
dist_age = dataset["age_distance"].astype(int)


def convert_JW(feature, breakpoints=[0.75, 0.88, 0.933], values=[3, 2, 1, 0]):
    i = bisect_left(breakpoints, feature)
    return values[i]


dist_fn = [convert_JW(feature) for feature in fn_feature]
dist_ln = [convert_JW(feature) for feature in ln_feature]

dataset_values = np.array([dist_age, dist_fn, dist_ln]).transpose()
# dataset_values = np.array([dist_fn, dist_ln]).transpose()


# TEST CLASS
print(f"starting CLASS")
em = ExpectationMaximization(dataset_values)
result = em.em_steps(1)
print(f"Theta_M: \n {result}")
# bayes = em.bayes_conversion(result)
# print(f"Bayes dist \n {bayes}")
