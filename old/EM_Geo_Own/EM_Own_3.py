import pandas as pd
import numpy as np
from scipy.optimize import minimize
from bisect import bisect_left

# Junget 1850-1845
# dataset = pd.read_csv(
#     "C:/thesis_code/Github/Experiments/data/junget_1850_1845")

# Junget 1860-1850
# dataset = pd.read_csv(
#     "C:/thesis_code/Github/Experiments/data/junget_1860_1850")

# Thy 1850-1845
# dataset = pd.read_csv(
#     "C:/thesis_code/Github/Experiments/data/thy_parishes_1850_1845")

# Thy 1860-1850
dataset = pd.read_csv(
    "C:/thesis_code/Github/Experiments/data/thy_parishes_1860_1850")


class ExpectationMaximization:
    def __init__(
        self,
        data,
    ):
        self.data = data
        # Feature names: A = age, FN = first name, LN = last name(patronym)
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

                w = (p_A_M+p_FN_M+p_LN_M) / \
                    ((p_A_M+p_FN_M+p_LN_M) +
                     (p_A_U+p_FN_U+p_LN_U))
                w_vector[i] = w

            # M-STEP

            x_M = np.array([self.P_A_M, self.P_FN_M, self.P_LN_M])
            x_U = np.array([self.P_A_U, self.P_FN_U, self.P_LN_U])

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
        return [self.P_A_M, self.P_FN_M, self.P_LN_M, self.P_A_U, self.P_FN_U, self.P_LN_U]

    def L_M(self, x, data, w_vector):
        log_p_a = np.log(x[0]+x[0]**2+x[0]**3)
        log_p_f = np.log(x[1]+x[1]**2+x[1]**3+x[1]**4)
        log_p_l = np.log(x[2]+x[2]**2+x[2]**3+x[2]**4)
        log_p = np.array([log_p_a, log_p_f, log_p_l])
        result_1 = (data*np.log(x)) - log_p
        result = np.sum(w_vector * np.sum(result_1, axis=1))
        return -result

    def L_U(self, x, data, w_vector):
        log_p_a = np.log(x[0]+x[0]**2+x[0]**3)
        log_p_f = np.log(x[1]+x[1]**2+x[1]**3+x[1]**4)
        log_p_l = np.log(x[2]+x[2]**2+x[2]**3+x[2]**4)
        log_p = np.array([log_p_a, log_p_f, log_p_l])
        data_reverse = np.array([3, 4, 4]) - data
        # data_reverse = np.array([3, 4, 4], size=(1, 3)) - data
        result_1 = data_reverse*np.log(x) - log_p
        result = np.sum((1-w_vector) * np.sum(result_1, axis=1))
        return -result

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

    def evaluation_bayes(self, data, results, data_b, path):
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
            prob = (P_A_M + P_F_M + P_L_M) / \
                ((P_A_M + P_F_M + P_L_M)+(P_A_U + P_F_U + P_L_U))
            probabilities.append('%.4f' % (prob))
        data_b['EM probabilities'] = probabilities
        data_b.to_csv(
            f"C:/thesis_code/Github/Experiments/results_after_EM/EM_Own_3/{path}", index=False)
        f = open(
            "C:/thesis_code/Github/Experiments/results_after_EM/EM_Own_3/p_values.txt", "a")
        f.write(f"Results from {path}: {results}\n")


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


# TEST CLASS
print(f"starting CLASS")
em = ExpectationMaximization(dataset_values)
results = em.em_steps(7)
print(f"Results: \n {results}")
em.evaluation_bayes(dataset_values, results,
                    data_bisect, "thy_parishes_1860_1850")
