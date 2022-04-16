from numpy.lib.npyio import savetxt
import pandas as pd
import numpy as np
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
        # # print(self.geo_dist_M)
        # self.geo_dist_U = self.create_geo_dist(
        #     self.geo_list_U, elements_dimensions)
        # self.distribution_spread = np.empty(elements_dimensions)
        # print(self.geo_dist_U)
        # self.geo_dist_M = self.geometric_dist(0.5)
        # self.geo_dist_U = self.geo_dist_M[::-1]

    def em_steps(self, iterations=100):
        for i in range(iterations):
            w_vector = np.zeros(len(self.data))
            # E-STEP
            t1 = pc()
            for i in range(len(self.data)):
                theta_value_M = self.theta_M[tuple(self.data[i])]
                theta_value_U = self.theta_U[tuple(self.data[i])]
                # geo_dist_M = self.geo_dist_M[tuple(self.data[i])]
                # geo_dist_U = self.geo_dist_U[tuple(self.data[i])]
                # self.distribution_spread[tuple(self.data[i])] += 1
                # w = (theta_value_M * geo_dist_M) / (
                #     (theta_value_M * geo_dist_M) + (
                #         theta_value_U * geo_dist_U)
                # )
                w = (theta_value_M * self.p_M) / (
                    (theta_value_M * self.p_M) + (theta_value_U * self.p_U)
                )
                w_vector[i] = w

            # M-STEP
            t2 = pc()
            # print(t2-t1)
            for i in range(len(self.data)):
                self.theta_M[tuple(self.data[i])] = (
                    self.theta_M[tuple(self.data[i])] + w_vector[i]
                )
                self.theta_U[tuple(self.data[i])] = self.theta_U[
                    tuple(self.data[i])
                ] + (1 - w_vector[i])

            # Normalize theta_M and theta_U
            self.theta_M = self.theta_M / self.theta_M.sum()
            self.theta_U = self.theta_U / self.theta_U.sum()

            self.p_M = np.mean(w_vector)
            self.p_U = 1 - self.p_M
        print(f"Theta_U \n {self.theta_U}")
        # print(f"Distribution spread: \n {self.distribution_spread}")
        return self.theta_M

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

    # def bisect(self, dataset, age_included=True):
    #     if age_included == True:
    #         pass

    def bayes_conversion(self, theta_M):
        dist = np.empty(self.elements_dimensions)
        for x, v in np.ndenumerate(dist):
            dist[x] = (theta_M[x]*self.p_M) / \
                ((theta_M[x]*self.p_M) +
                 (self.theta_U[x]*self.p_U))
        # b = np.linalg.norm(dist)
        # return dist/b
        return dist


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
em = ExpectationMaximization(dataset_values, 100, 3, [3, 4, 4])
result = em.em_steps(10)
print(f"Theta_M: \n {result}")
bayes = em.bayes_conversion(result)
print(f"Bayes dist \n {bayes}")
