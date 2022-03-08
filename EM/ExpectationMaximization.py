from numpy.lib.npyio import savetxt
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from bisect import bisect_left
from time import perf_counter as pc


# dataset = pd.read_csv(
#     "C:/thesis_code/Github/data/comp_sets/thy_parishes_1850_1845")
dataset = pd.read_csv("C:/thesis_code/Github/data/comp_sets/junget_1850_1845")
# dataset = pd.read_csv("C:/thesis_code/Github/data/comp_sets/testset")
# dataset = pd.read_csv("C:/thesis_code/Github/data/comp_sets/testset_2d")


class ExpectationMaximization:
    def __init__(
        self,
        data,
        match_guess,
    ):
        self.data = data
        self.P_A_M = self.P_FN_M = self.P_LN_M = 0.5
        self.P_A_U = self.P_FN_U = self.P_LN_U = 0.5
        self.p_M = match_guess / len(data)
        print(self.p_M)
        self.p_U = 1 - self.p_M

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

                w = ((p_A_M*p_FN_M*p_LN_M)*self.p_M) / \
                    (((p_A_M*p_FN_M*p_LN_M)*self.p_M) +
                     ((p_A_U*p_FN_U*p_LN_U)*self.p_U))
                w_vector[i] = w

            # M-STEP

            self.p_M = np.mean(w_vector)
            self.p_U = 1 - self.p_M

            # x_M = [self.P_A_M, self.P_FN_M, self.P_LN_M]
            # x_U = [self.P_A_U, self.P_FN_U, self.P_LN_U]
            x_M = np.array([self.P_A_M, self.P_FN_M, self.P_LN_M])
            x_U = np.array([self.P_A_U, self.P_FN_U, self.P_LN_U])

            #log_p = np.array([log_p_a, log_p_f, log_p_l])

            res_M = minimize(self.L_M, x0=x_M, args=(self.data, w_vector),
                             method='L-BFGS-B', bounds=[(0.001, 1), (0.001, 1), (0.001, 1)])
            res_U = minimize(self.L_U, x0=x_U, args=(self.data, w_vector),
                             method='L-BFGS-B', bounds=[(0.001, 1), (0.001, 1), (0.001, 1)])

            print(f"x for matches: {res_M.x}")
            print(f"x for non-matches: {res_U.x}")

            self.P_A_M = res_M.x[0]
            self.P_FN_M = res_M.x[1]
            self.P_LN_M = res_M.x[2]

            self.P_A_U = res_U.x[0]
            self.P_FN_U = res_U.x[1]
            self.P_LN_U = res_U.x[2]
        print(f"p_M: {self.p_M} \n p_U: {self.p_U}")
        return [self.P_A_M, self.P_FN_M, self.P_LN_M, self.P_A_U, self.P_FN_U, self.P_LN_U]

    # def L_M(self, x, data, w_vector):
    #     log_p_a = np.log(x[0]+x[0]**2+x[0]**3)
    #     log_p_f = np.log(x[1]+x[1]**2+x[1]**3+x[1]**4)
    #     log_p_l = np.log(x[2]+x[2]**2+x[2]**3+x[2]**4)
    #     sum = 0
    #     for i in range(len(data)):
    #         sum += w_vector[i]*(data[i][0]*np.log(x[0])-log_p_a+data[i][1]
    #                             * np.log(x[1])-log_p_f+data[i][2]*np.log(x[2])-log_p_l)
    #     return -1*sum

    def L_U(self, x, data, w_vector):
        log_p_a = np.log(x[0]+x[0]**2+x[0]**3)
        log_p_f = np.log(x[1]+x[1]**2+x[1]**3+x[1]**4)
        log_p_l = np.log(x[2]+x[2]**2+x[2]**3+x[2]**4)
        sum = 0
        for i in range(len(data)):
            sum += (1-w_vector[i])*((3-data[i][0])*np.log(x[0])-log_p_a+(4-data[i][1])
                                    * np.log(x[1])-log_p_f+(4-data[i][2])*np.log(x[2])-log_p_l)
        return -1*sum

    def L_M(self, x, data, w_vector):
        # brug disse som argumenter i minimize i stedet - muligvis også disse: np_log_a = np.log(x[0])
        log_p_a = np.log(x[0]+x[0]**2+x[0]**3)
        log_p_f = np.log(x[1]+x[1]**2+x[1]**3+x[1]**4)
        log_p_l = np.log(x[2]+x[2]**2+x[2]**3+x[2]**4)
        result_1 = data*np.log(x)
        result_1[:, 0] = result_1[:, 0]-log_p_a
        result_1[:, 1] = result_1[:, 1]-log_p_f
        result_1[:, 2] = result_1[:, 2]-log_p_l
        result = np.sum(w_vector * np.sum(result_1, axis=1))
        return -result

    # def L_U(self, x, data, w_vector):
    #     # brug disse som argumenter i minimize i stedet - muligvis også disse: np_log_a = np.log(x[0])
    #     log_p_a = np.log(x[0]+x[0]**2+x[0]**3)
    #     log_p_f = np.log(x[1]+x[1]**2+x[1]**3+x[1]**4)
    #     log_p_l = np.log(x[2]+x[2]**2+x[2]**3+x[2]**4)   #
    #     result_1 = data*np.log(x)
    #     result_1[:, 0] = result_1[:, 0]-log_p_a
    #     result_1[:, 1] = result_1[:, 1]-log_p_f
    #     result_1[:, 2] = result_1[:, 2]-log_p_l    #
    #     result = np.sum((1-w_vector) * np.sum(result_1, axis=1))
    #     return -result

    # def L_M(self, x, data, w_vector):
    #     length = len(data)
    #     log_p_a = np.log(x[0]+x[0]**2+x[0]**3)
    #     log_p_f = np.log(x[1]+x[1]**2+x[1]**3+x[1]**4)
    #     log_p_l = np.log(x[2]+x[2]**2+x[2]**3+x[2]**4)
    #     a_i = data[np.arange(length)][0].sum()
    #     f_i = data[np.arange(length)][1].sum()
    #     l_i = data[np.arange(length)][2].sum()
    #     sum = (w_vector[np.arange(length)].sum())*(a_i*np.log(x[0])-log_p_a+f_i *
    #                                                np.log(x[1])-log_p_f+l_i*np.log(x[2])-log_p_l)
    #     return -1*sum

    # def L_U(self, x, data, w_vector):
    #     length = len(data)
    #     log_p_a = np.log(x[0]+x[0]**2+x[0]**3)
    #     log_p_f = np.log(x[1]+x[1]**2+x[1]**3+x[1]**4)
    #     log_p_l = np.log(x[2]+x[2]**2+x[2]**3+x[2]**4)
    #     a_i = data[np.arange(length)][0].sum()
    #     f_i = data[np.arange(length)][1].sum()
    #     l_i = data[np.arange(length)][2].sum()
    #     sum = (w_vector[np.arange(length)].sum())*((3-a_i)*np.log(x[0])-log_p_a+(4-f_i) *
    #                                                np.log(x[1])-log_p_f+(4-l_i)*np.log(x[2])-log_p_l)
    #     return -1*sum

    # function to calculate geometric distribution in the E-step
    # p is the given probability, k is the feature value

    def geo_dist_names(self, p, k, matches=True):
        if not matches:
            k = 3 - k
        return (p**(1+k))/(p+p**2+p**3+p**4)

    def geo_dist_age(self, p, k, matches=True):
        if not matches:
            k = 2 - k
        return (p**(1+k))/(p+p**2+p**3)

    def evaluation_bayes(self, data, results, data_b):
        probabilities = []
        for i in range(len(data)):
            P_A_M = self.geo_dist_age(results[0], data[i][0])
            P_F_M = self.geo_dist_names(results[1], data[i][1])
            P_L_M = self.geo_dist_names(results[2], data[i][2])
            P_A_U = self.geo_dist_age(
                results[3], data[i][0], matches=False)
            P_F_U = self.geo_dist_names(
                results[4], data[i][1], matches=False)
            P_L_U = self.geo_dist_names(
                results[5], data[i][2], matches=False)
            prob = ((P_A_M * P_F_M * P_L_M)*self.p_M) / \
                (((P_A_M * P_F_M * P_L_M)*self.p_M) +
                 ((P_A_U * P_F_U * P_L_U)*self.p_U))
            # prob = (P_A_M * P_F_M * P_L_M) / \
            #     ((P_A_M * P_F_M * P_L_M)+(P_A_U * P_F_U * P_L_U))
            # prob = (P_A_M + P_F_M + P_L_M) / \
            #     ((P_A_M + P_F_M + P_L_M)+(P_A_U + P_F_U + P_L_U))
            probabilities.append('%.4f' % (prob))
        data_b['EM probabilities'] = probabilities
        data_b.to_csv(
            "C:/thesis_code/Github/data/results_EM_Geo/thy_parishes_1850_1845_EM_Geo_results", index=False)


fn_feature = dataset["fn_score"]
ln_feature = dataset["ln_score"]
dist_age = dataset["age_distance"].astype(int)


def convert_JW(feature, breakpoints=[0.75, 0.88, 0.933], values=[3, 2, 1, 0]):
    i = bisect_left(breakpoints, feature)
    return values[i]


dist_fn = [convert_JW(feature) for feature in fn_feature]
dist_ln = [convert_JW(feature) for feature in ln_feature]

data_bisect = pd.DataFrame(
    {'age': dist_age, 'first name': dist_fn, 'last_name': dist_ln})

dataset_values = np.array([dist_age, dist_fn, dist_ln]).transpose()
# dataset_values = np.array([dist_fn, dist_ln]).transpose()


# TEST CLASS
print(f"starting CLASS")
em = ExpectationMaximization(dataset_values, 100)
results = em.em_steps(7)
# results = [0.001, 0.50860069, 0.001, 0.71309623, 0.21479873, 0.30512826]
print(f"Results: \n {results}")
#em.evaluation_bayes(dataset_values, results, data_bisect)
# print(f"Probability results: \n {bayes}")
