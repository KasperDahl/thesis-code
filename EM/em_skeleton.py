from numpy.lib.npyio import savetxt
import pandas as pd
import numpy as np
from bisect import bisect_left

dataset = pd.read_csv("C:/thesis_code/Github/data//comp_sets/junget_1850_1845")
fn_feature = dataset['fn_score']
ln_feature = dataset['ln_score']
# not entirely sure why this list is received as floats
dist_age = dataset['age_distance'].astype(int)
# 1. Distance Bins

# The string numbers below are based on research by Winkler 1988


def convert_JW(feature, breakpoints=[0.75, 0.88, 0.933], values=[3, 2, 1, 0]):
    i = bisect_left(breakpoints, feature)
    return values[i]


dist_fn = [convert_JW(feature) for feature in fn_feature]
dist_ln = [convert_JW(feature) for feature in ln_feature]


# Create numpy array with values for use in E-step
dataset_values = np.array([dist_age, dist_fn, dist_ln]).transpose()

# 2. Parameters

# 3-dimensional array consisting of the multinomial distance bins (fn, ln, age)
# LOOK INTO: how to automate this (could be min-max range calculation, or convert to sets - otherwise just received as parameters)
dimensions = 3*4*4
theta_M = np.full((3, 4, 4), 1/dimensions)
theta_U = np.full((3, 4, 4), 1/dimensions)

# probability of being a match - should be a reasonable guess
# example: if I estimate a 100 matches in the comparison space of two censuses of a 1000 records each
# the estimated probability is 100/(1000 X 1000) = 0,0001
match_guess = 100
p_M = match_guess/(len(dataset))
p_U = 1 - p_M

# 3. Loop over steps E and M


def em_steps(p_M, p_U, theta_M, theta_U):

    # E-STEP
    # initialize loop - number of iterations will be changed later
    # loop until convergence - or maximum 100 iterations (or other number)
    # if the latter - this information should be outputted
    n_iterations = 20
    for i in range(n_iterations):
        w = np.zeros(len(dataset_values))

        #print("start E-step")
        # get features and look up corresponding theta element for pair in np.nditer(dataset_values):
        for i in range(len(dataset_values)):
            distances = dataset_values[i]
            # LOOK INTO: Maybe I can use unpacking, (*distances), and pass as args that way. Mosh: 5,22
            theta_value_M = theta_M[distances[0], distances[1], distances[2]]
            theta_value_U = theta_U[distances[0], distances[1], distances[2]]
            # vector w
            w_vector = (theta_value_M*p_M) / \
                ((theta_value_M*p_M)+(theta_value_U*p_U))
            w[i] = w_vector

        #print("end E-step")
        # convert to Numpy-array
        w_np = np.array(w)

        # M-STEP
        #print("start M-step")
        # maximize theta values in 3-d arrays
        for i in range(len(dataset_values)):
            distances = dataset_values[i]
            theta_M[distances[0], distances[1], distances[2]
                    ] = theta_M[distances[0], distances[1], distances[2]] + w_np[i]
            theta_U[distances[0], distances[1], distances[2]
                    ] = theta_U[distances[0], distances[1], distances[2]] + (1-w_np[i])

        # Normalize theta_M and theta_U
        theta_M = theta_M/theta_M.sum(keepdims=True)
        theta_U = theta_U/theta_U.sum(keepdims=True)

        p_M = np.mean(w_np)
        p_U = 1 - p_M

        #print(f"P_M: {p_M}, P_U: {p_U}")

    return theta_M


print(f"starting loop")
theta_final = em_steps(p_M, p_U, theta_M, theta_U)
print(theta_final)

#reshape = theta_final.reshape(theta_final.shape[0], -1)
#savetxt('C:/thesis_code/Github/data/numpy_arrays/np_array', reshape)
np.save('C:/thesis_code/Github/data/numpy_arrays/np_array1.npy', theta_final)
