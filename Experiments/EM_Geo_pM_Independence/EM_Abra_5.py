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
# dataset = pd.read_csv(
#     "C:/thesis_code/Github/Experiments/data/thy_parishes_1860_1850")

# Manual 1850-1845
# dataset = pd.read_csv(
#     "C:/thesis_code/Github/Experiments/data/manual_1850_1845")

# Manual 1860-1850
dataset = pd.read_csv(
    "C:/thesis_code/Github/Experiments/data/manual_1860_1850")


class ExpectationMaximization:
    def __init__(
        self,
        data,
        match_guess,
    ):
        self.data = data
        # Feature names: A = age, FN = first name, LN = last name(patronym), FAMN = family name, BP = birth parish
        self.P_A_M = self.P_FN_M = self.P_LN_M = self.P_FAMN_M = self.P_BP_M = 0.5
        self.P_A_U = self.P_FN_U = self.P_LN_U = self.P_FAMN_U = self.P_BP_U = 0.5
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
                p_FAMN_M = self.geo_dist_names(
                    self.P_FAMN_M, self.data[i][3])
                p_FAMN_U = self.geo_dist_names(
                    self.P_FAMN_U, self.data[i][3], matches=False)
                p_BP_M = self.geo_dist_names(
                    self.P_BP_M, self.data[i][4])
                p_BP_U = self.geo_dist_names(
                    self.P_BP_U, self.data[i][4], matches=False)

                w = ((p_A_M*p_FN_M*p_LN_M*p_FAMN_M*p_BP_M)*self.p_M) / \
                    (((p_A_M*p_FN_M*p_LN_M*p_FAMN_M*p_BP_M)*self.p_M) +
                     ((p_A_U*p_FN_U*p_LN_U*p_FAMN_U*p_BP_U)*self.p_U))
                w_vector[i] = w

            # M-STEP

            self.p_M = np.mean(w_vector)
            self.p_U = 1 - self.p_M

            x_M = np.array(
                [self.P_A_M, self.P_FN_M, self.P_LN_M, self.P_FAMN_M, self.P_BP_M])
            x_U = np.array(
                [self.P_A_U, self.P_FN_U, self.P_LN_U, self.P_FAMN_U, self.P_BP_U])

            res_M = minimize(self.L_M, x0=x_M, args=(self.data, w_vector),
                             method='L-BFGS-B', bounds=[(0.001, 1), (0.001, 1), (0.001, 1), (0.001, 1), (0.001, 1)])
            res_U = minimize(self.L_U, x0=x_U, args=(self.data, w_vector),
                             method='L-BFGS-B', bounds=[(0.001, 1), (0.001, 1), (0.001, 1), (0.001, 1), (0.001, 1)])

            print(f"x for matches: {res_M.x}")
            print(f"x for non-matches: {res_U.x}")

            self.P_A_M = res_M.x[0]
            self.P_FN_M = res_M.x[1]
            self.P_LN_M = res_M.x[2]
            self.P_FAMN_M = res_M.x[3]
            self.P_BP_M = res_M.x[4]

            self.P_A_U = res_U.x[0]
            self.P_FN_U = res_U.x[1]
            self.P_LN_U = res_U.x[2]
            self.P_FAMN_U = res_U.x[3]
            self.P_BP_U = res_U.x[4]
        print(f"p_M: {self.p_M}")
        return [self.P_A_M, self.P_FN_M, self.P_LN_M, self.P_FAMN_M, self.P_BP_M, self.P_A_U, self.P_FN_U, self.P_LN_U, self.P_FAMN_U, self.P_BP_U]

    def L_M(self, x, data, w_vector):
        log_p_a = np.log(x[0]+x[0]**2+x[0]**3)
        log_p_f = np.log(x[1]+x[1]**2+x[1]**3+x[1]**4)
        log_p_l = np.log(x[2]+x[2]**2+x[2]**3+x[2]**4)
        log_p_fam = np.log(x[3]+x[3]**2+x[3]**3+x[3]**4)
        log_p_b = np.log(x[4]+x[4]**2+x[4]**3+x[4]**4)
        log_p = np.array([log_p_a, log_p_f, log_p_l, log_p_fam, log_p_b])
        result_1 = (data*np.log(x)) - log_p
        result = np.sum(w_vector * np.sum(result_1, axis=1))
        return -result

    def L_U(self, x, data, w_vector):
        log_p_a = np.log(x[0]+x[0]**2+x[0]**3)
        log_p_f = np.log(x[1]+x[1]**2+x[1]**3+x[1]**4)
        log_p_l = np.log(x[2]+x[2]**2+x[2]**3+x[2]**4)
        log_p_fam = np.log(x[3]+x[3]**2+x[3]**3+x[3]**4)
        log_p_b = np.log(x[4]+x[4]**2+x[4]**3+x[4]**4)
        log_p = np.array([log_p_a, log_p_f, log_p_l, log_p_fam, log_p_b])
        data_reverse = np.array([3, 4, 4, 4, 4]) - data
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
            P_FAM_M = self.geo_dist_names(results[3], data[i][3])
            P_BP_M = self.geo_dist_names(results[4], data[i][4])
            P_A_U = self.geo_dist_age(
                results[5], data[i][0], matches=False)
            P_F_U = self.geo_dist_names(
                results[6], data[i][1], matches=False)
            P_L_U = self.geo_dist_names(
                results[7], data[i][2], matches=False)
            P_FAM_U = self.geo_dist_names(
                results[8], data[i][3], matches=False)
            P_BP_U = self.geo_dist_names(
                results[9], data[i][4], matches=False)
            prob = ((P_A_M * P_F_M * P_L_M * P_FAM_M * P_BP_M)*self.p_M) / (((P_A_M * P_F_M * P_L_M *
                                                                              P_FAM_M * P_BP_M)*self.p_M) + ((P_A_U * P_F_U * P_L_U * P_FAM_U * P_BP_U)*self.p_U))
            probabilities.append('%.4f' % (prob))
        data_b['EM probabilities'] = probabilities
        data_b.to_csv(
            f"C:/thesis_code/Github/Experiments/results_after_EM/EM_Abra_5/{path}", index=False)
        f = open(
            "C:/thesis_code/Github/Experiments/results_after_EM/EM_Abra_5/p_values.txt", "a")
        f.write(f"Results from {path}: {results}\n")


fn_feature = dataset["fn_score"]
ln_feature = dataset["ln_score"]
famn_feature = dataset["fam_n_score"]
bp_feature = dataset["bp_score"]
dist_age = dataset["age_distance"].astype(int)


def convert_JW(feature, breakpoints=[0.75, 0.88, 0.933], values=[3, 2, 1, 0]):
    i = bisect_left(breakpoints, feature)
    return values[i]


dist_fn = [convert_JW(feature) for feature in fn_feature]
dist_ln = [convert_JW(feature) for feature in ln_feature]
dist_famn = [convert_JW(feature) for feature in famn_feature]
dist_bp = [convert_JW(feature) for feature in bp_feature]

data_bisect = pd.DataFrame(
    {'age': dist_age, 'first name': dist_fn, 'last_name': dist_ln, 'family_name': dist_famn, 'birth_parish': dist_bp})

dataset_values = np.array(
    [dist_age, dist_fn, dist_ln, dist_famn, dist_bp]).transpose()


# TEST CLASS
print(f"starting CLASS")
em = ExpectationMaximization(dataset_values, 9600)
results = em.em_steps(7)
print(f"Results: \n {results}")
em.evaluation_bayes(dataset_values, results,
                    data_bisect, "manual_1860_1850")
