from numpy.lib.npyio import savetxt
import pandas as pd
import numpy as np
from bisect import bisect_left
from time import perf_counter as pc


dataset = pd.read_csv(
    "C:/thesis_code/Github/data/comp_sets/thy_parishes_1850_1845")
#dataset = pd.read_csv("C:/thesis_code/Github/data/comp_sets/junget_1850_1845")


class ExpectationMaximization:
    def __init__(
        self,
        data,
        match_guess,
    ):
        self.data = data
        self.p_M = match_guess / len(data)
        self.p_U = 1 - self.p_M

        # Priors are set equal to 1/k for the unmatched and to 1/k + ((k-1)/2 - (i-1))/(k^2) for the matched
        self.theta_A_M = np.zeros(3)
        self.theta_FN_M = self.theta_P_M = np.zeros(4)
        for i in range(len(self.theta_A_M)):
            self.theta_A_M[i] = 1/len(self.theta_A_M) + \
                ((len(self.theta_A_M)-1)/2-(i-1))/(len(self.theta_A_M)**2)
        for i in range(len(self.theta_FN_M)):
            self.theta_FN_M[i] = 1/len(self.theta_FN_M) + (
                (len(self.theta_FN_M)-1)/2-(i-1))/(len(self.theta_FN_M)**2)
        for i in range(len(self.theta_P_M)):
            self.theta_P_M[i] = 1/len(self.theta_P_M) + \
                ((len(self.theta_P_M)-1)/2-(i-1))/(len(self.theta_P_M)**2)
        self.theta_A_U = np.full(3, 1/3)
        self.theta_FN_U = self.theta_P_U = np.full(4, 1/4)

        # print(self.theta_A_M, self.theta_FN_M, self.theta_P_M)
        # print(self.theta_A_U, self.theta_FN_U, self.theta_P_U)

    def em_steps(self, iterations=100):
        for i in range(iterations):
            w_vector = np.zeros(len(self.data))
            # E-STEP
            for i in range(len(self.data)):
                p_A_M = self.theta_A_M[self.data[i][0]]
                p_A_U = self.theta_A_U[self.data[i][0]]

                p_FN_M = self.theta_FN_M[self.data[i][1]]
                p_FN_U = self.theta_FN_U[self.data[i][1]]

                p_P_M = self.theta_P_M[self.data[i][2]]
                p_P_U = self.theta_P_U[self.data[i][2]]

                w = (p_A_M*p_FN_M*p_P_M*self.p_M)/((p_A_M*p_FN_M *
                                                    p_P_M*self.p_M) + (p_A_U*p_FN_U*p_P_U*self.p_U))

                w_vector[i] = w
            # print(len(w_vector))
            # print(w_vector)
            # M-STEP

            self.p_M = np.mean(w_vector)
            self.p_U = 1 - self.p_M

            for i in range(len(self.data)):
                self.theta_A_M[self.data[i][0]
                               ] = self.theta_A_M[self.data[i][0]] + w_vector[i]
                self.theta_A_U[self.data[i][0]
                               ] = self.theta_A_U[self.data[i][0]] + (1-w_vector[i])

                self.theta_FN_M[self.data[i][1]
                                ] = self.theta_FN_M[self.data[i][1]] + w_vector[i]
                self.theta_FN_U[self.data[i][1]
                                ] = self.theta_FN_U[self.data[i][1]] + (1-w_vector[i])

                self.theta_P_M[self.data[i][2]
                               ] = self.theta_P_M[self.data[i][2]] + w_vector[i]
                self.theta_P_U[self.data[i][2]
                               ] = self.theta_P_U[self.data[i][2]] + (1-w_vector[i])

            self.theta_A_M = self.theta_A_M / self.theta_A_M.sum()
            self.theta_A_U = self.theta_A_U / self.theta_A_U.sum()

            self.theta_FN_M = self.theta_FN_M / self.theta_FN_M.sum()
            self.theta_FN_U = self.theta_FN_U / self.theta_FN_U.sum()

            self.theta_P_M = self.theta_P_M / self.theta_P_M.sum()
            self.theta_P_U = self.theta_P_U / self.theta_P_U.sum()

        # print(self.theta_A_M)
        # print(self.theta_A_U)

        # print(self.theta_FN_M)
        # print(self.theta_FN_U)

        # print(self.theta_P_M)
        # print(self.theta_P_U)

    def bayes_conversion(self, data_b, path):
        probabilities = []
        for i in range(len(self.data)):
            p_A_M = self.theta_A_M[self.data[i][0]]
            p_A_U = self.theta_A_U[self.data[i][0]]

            p_FN_M = self.theta_FN_M[self.data[i][1]]
            p_FN_U = self.theta_FN_U[self.data[i][1]]

            p_P_M = self.theta_P_M[self.data[i][2]]
            p_P_U = self.theta_P_U[self.data[i][2]]

            prob = (p_A_M*p_FN_M*p_P_M*self.p_M)/((p_A_M*p_FN_M *
                                                   p_P_M*self.p_M) + (p_A_U*p_FN_U*p_P_U*self.p_U))
            probabilities.append('%.4f' % (prob))
        data_b['EM probabilities'] = probabilities
        data_b.to_csv(
            f"C:/thesis_code/Github/Experiments/results_after_EM/Abramitzsky/{path}", index=False)
        # print(probabilities)


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
data_bisect = pd.DataFrame(
    {'age': dist_age, 'first name': dist_fn, 'last_name': dist_ln})


# TEST CLASS
print(f"starting CLASS")
em = ExpectationMaximization(dataset_values, 2200)
em.em_steps(50)
probs = em.bayes_conversion(data_bisect, "thy_1850_1845")
# print(probs)
