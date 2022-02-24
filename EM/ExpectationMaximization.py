from numpy.lib.npyio import savetxt
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from bisect import bisect_left
from time import perf_counter as pc


dataset = pd.read_csv(
    "C:/thesis_code/Github/data/comp_sets/thy_parishes_1850_1845")
# dataset = pd.read_csv("C:/thesis_code/Github/data/comp_sets/junget_1850_1845")
# dataset = pd.read_csv("C:/thesis_code/Github/data/comp_sets/testset")
# dataset = pd.read_csv("C:/thesis_code/Github/data/comp_sets/testset_2d")


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
        self.elements_dimensions = elements_dimensions
        self.elements_product = np.prod(elements_dimensions)
        self.theta_M = np.full(elements_dimensions, 1 / self.elements_product)
        self.theta_U = np.full(elements_dimensions, 1 / self.elements_product)
        self.p_M = match_guess / len(data)
        self.p_U = 1 - self.p_M
        # creation of geometric distribution arrays for both clusters
        # self.geo_list_M = self.geometric_dist(0.95)
        # self.geo_list_U = self.geo_list_M[0:max(
        #     elements_dimensions)][::-1]
        # self.geo_dist_M = self.create_geo_dist(
        #     self.geo_list_M, elements_dimensions)
        # self.geo_dist_U = self.create_geo_dist(
        #     self.geo_list_U, elements_dimensions)

        # NEW THETA ARRAYS - A=Age, FN=First name, LN=Last name
        self.theta_A_M = np.full(3, 1/elements_dimensions[0])
        self.theta_A_U = np.full(3, 1/elements_dimensions[0])
        self.theta_FN_M = np.full(4, 1/elements_dimensions[1])
        self.theta_FN_U = np.full(4, 1/elements_dimensions[1])
        self.theta_LN_M = np.full(4, 1/elements_dimensions[2])
        self.theta_LN_U = np.full(4, 1/elements_dimensions[2])

        # GEOMETRIC DISTRIBUTIONS
        self.p = 0.95
        self.geo_age_M = [self.p, (1-self.p)*self.p, (1-self.p)**2]
        self.geo_age_U = self.geo_age_M[::-1]
        self.geo_names_M = [
            self.p, (1-self.p)*self.p, (1-self.p)**2*self.p, (1-self.p)**3]
        self.geo_names_U = self.geo_names_M[::-1]

    def em_steps(self, iterations=100):
        for i in range(iterations):
            w_vector = np.zeros(len(self.data))
            # E-STEP
            for i in range(len(self.data)):
                # The three feature values from the data are split and for each of them the associated values
                # from the respective theta array are found and multiplied with the respective geometric distribution value;
                # these variables are then used to find w for those given feature values
                p_A_M = self.theta_A_M[self.data[i][0]] * \
                    self.geo_age_M[self.data[i][0]]
                p_A_U = self.theta_A_U[self.data[i][0]] * \
                    self.geo_age_U[self.data[i][0]]
                p_FN_M = self.theta_FN_M[self.data[i]
                                         [1]]*self.geo_names_M[self.data[i][1]]
                p_FN_U = self.theta_FN_U[self.data[i]
                                         [1]]*self.geo_names_U[self.data[i][1]]
                p_LN_M = self.theta_LN_M[self.data[i]
                                         [2]]*self.geo_names_M[self.data[i][2]]
                p_LN_U = self.theta_LN_U[self.data[i]
                                         [2]]*self.geo_names_U[self.data[i][2]]

                w = (p_A_M*p_FN_M*p_LN_M) / \
                    ((p_A_M*p_FN_M*p_LN_M)+(p_A_U*p_FN_U*p_LN_U))
                w_vector[i] = w

                # theta_value_M = self.theta_M[tuple(self.data[i])]
                # theta_value_U = self.theta_U[tuple(self.data[i])]
                # geo_dist_M = self.geo_dist_M[tuple(self.data[i])]
                # geo_dist_U = self.geo_dist_U[tuple(self.data[i])]
                # w = (theta_value_M * geo_dist_M) / (
                #     (theta_value_M * geo_dist_M) + (
                #         theta_value_U * geo_dist_U)
                # )
                # w_vector[i] = w
            # print(w_vector)
            # M-STEP
            for i in range(len(self.data)):
                theta_A_M_1 = self.theta_A_M[self.data[i][0]]
                theta_A_U_1 = self.theta_A_U[self.data[i][0]]
                theta_FN_M_1 = self.theta_FN_M[self.data[i][1]]
                theta_FN_U_1 = self.theta_FN_U[self.data[i][1]]
                theta_LN_M_1 = self.theta_LN_M[self.data[i][2]]
                theta_LN_U_1 = self.theta_LN_U[self.data[i][2]]
                L = minimize(self.obj_func_L, x0=6, args=(
                    theta_A_M_1, theta_A_U_1, theta_FN_M_1, theta_FN_U_1, theta_LN_M_1),  method='L-BFGS-B')
                #     self.theta_M[tuple(self.data[i])] = (
                #         self.theta_M[tuple(self.data[i])] + w_vector[i]
                #     )
                #     self.theta_U[tuple(self.data[i])] = self.theta_U[
                #         tuple(self.data[i])
                #     ] + (1 - w_vector[i])

                # # Normalize theta_M and theta_U
                # self.theta_M = self.theta_M / self.theta_M.sum()
                # self.theta_U = self.theta_U / self.theta_U.sum()

        return 1

    def obj_func_L(self, A_M, A_U, FN_M, FN_U, LN_M, LN_U):
        return A_M+(1-A_U)+FN_M+(1-FN_U)+LN_M+(1-LN_U)

    def bayes_conversion(self, theta_M):
        dist = np.empty(self.elements_dimensions)
        length = len(self.data)
        for x, v in np.ndenumerate(dist):
            dist[x] = (theta_M[x]*self.geo_dist_M[x]) / \
                ((theta_M[x]*self.geo_dist_M[x]) +
                 (self.theta_U[x]*self.geo_dist_U[x]))
        return dist

    # an issue with the geometric list of non-matches, geo_list_U, is that different values of element_dimensions-variable
    # will affect the creation since the reversed list is based on the value of highest element in the list below
    # The list below is also not generalized
    # def geometric_dist(self, p):
    #     list = [p, (1-p)*p, (1-p)**2*p, (1-p)**3*p]
    #     return list

    # def create_geo_dist(self, geo_list, elements_dimensions):
    #     geo_array = np.ones(elements_dimensions)
    #     it = np.nditer(geo_array, flags=['multi_index'])
    #     for x in it:
    #         for y in it.multi_index:
    #             geo_array[it.multi_index] = geo_array[it.multi_index] * geo_list[y]
    #     return geo_array

    # def weighted_p(self, tuple, geo_dist):
    #     a, b = tuple
    #     p = geo_dist[a] * geo_dist[b]
    #     return p


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
em = ExpectationMaximization(dataset_values, 2200, 3, [3, 4, 4])
result = em.em_steps(10)
print(f"Theta_M: \n {result}")
# bayes = em.bayes_conversion(result)
# print(f"Bayes dist \n {bayes}")
