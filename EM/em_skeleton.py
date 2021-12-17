import pandas as pd
import numpy as np


dataset = pd.read_csv("C:/thesis_code/Github/data//comp_sets/junget_1850_1845")
fn_feature = dataset['fn_score']
ln_feature = dataset['ln_score']
# not entirely sure why this list is received as floats
dist_age = dataset['age_distance'].astype(int)
# 1. Distance Bins

# The string numbers below are based on research by Winkler 1988
# LOOK INTO: Christian suggest a better way: https://docs.python.org/3/library/bisect.html
# Also this might already be done in the classification step and then received like this
dist_fn = []
for name in fn_feature:
    if name >= 0.933:
        name = 0
    elif 0.933 >= name > 0.88:
        name = 1
    elif 0.88 >= name > 0.75:
        name = 2
    elif 0.75 >= name:
        name = 3
    dist_fn.append(name)

dist_ln = []
for name in ln_feature:
    if name >= 0.933:
        name = 0
    elif 0.933 >= name > 0.88:
        name = 1
    elif 0.88 >= name > 0.75:
        name = 2
    elif 0.75 >= name:
        name = 3
    dist_ln.append(name)

# Create numpy array with values for use in E-step
dataset_values = np.array([dist_age, dist_fn, dist_ln]).transpose()
# print(dataset_values[:90])

# 2. Parameters

# 3-dimensional array consisting of the multinomial distance bins (fn, ln, age)
# LOOK INTO: how to automate this (could be min-max range calculation, or convert to sets - otherwise just received as parameters)
dimensions = 3*4*4
theta_M = np.full((3, 4, 4), 1/dimensions)
theta_U = np.full((3, 4, 4), 1/dimensions)

# probability of being a match - should be a reasonable guess
# if I estimate a 100 matches in the comparison space of two censuses of a 1000 records each
# the estimated probability is 100/(1000 X 1000) = 0,0001
# QUESTION: the denominator below is the given comparison space (after blocking on gender and age +-2) -
# is it correct to use that and not the full comparison space of census_1*census_2
match_guess = 100
p_M = match_guess/(len(dataset))
p_U = 1 - p_M
print(p_M)

# 3. Loop over steps E and M


def em_steps(p_M, p_U, theta_M, theta_U):

    # initialize loop - number of iterations will be changed later
    # loop until convergence - or maximum 100 iterations (or other number)
    # if the latter - this information should be outputted
    n_iterations = 10
    for i in range(n_iterations):
        #w = np.zeros(len(dataset_values))
        w = []
        index = 0
        # E-STEP
        #print("start E-step")
        # get features and look up corresponding theta element
        # for pair in np.nditer(dataset_values):
        for i in range(len(dataset_values)):
            distances = dataset_values[i]
            # LOOK INTO: Maybe I can use unpacking, (*distances), and pass as args that way. Mosh: 5,22
            theta_value_M = theta_M[distances[0], distances[1], distances[2]]
            theta_value_U = theta_U[distances[0], distances[1], distances[2]]
            # vector w
            # LOOK INTO: indexing might be faster than appending
            w_vector = (theta_value_M*p_M) / \
                ((theta_value_M*p_M)+(theta_value_U*p_U))
            #w[index] = w_vector
            w.append(w_vector)
            index += 1
            # print(
            #    f"iteration: {i} ; w-vector: {w} ; theta_M: {theta_value_M} ; theta_U: {theta_value_U}")
        #print("end E-step")
        # convert to Numpy-array
        w_np = np.array(w)
        # print(w_np)
        # M-STEP
        #print("start M-step")
        # calculate the weighted
        # print(theta_U)
        for i in range(len(dataset_values)):
            distances = dataset_values[i]
            theta_M[distances[0], distances[1], distances[2]
                    ] = theta_M[distances[0], distances[1], distances[2]] + w_np[i]
            theta_U[distances[0], distances[1], distances[2]
                    ] = theta_U[distances[0], distances[1], distances[2]] + (1-w_np[i])

        # Normalize theta_M and theta_U
        theta_M = theta_M/theta_M.sum(keepdims=True)
        theta_U = theta_U/theta_U.sum(keepdims=True)
        """ print(np.sum(theta_M))
        print(theta_M)
        print(np.sum(theta_U))
        print(theta_U)
 """

        p_M = np.mean(w_np)
        print(p_M)
        # print(p_M_1)
        p_U = 1 - p_M


print(f"starting loop")
em_steps(p_M, p_U, theta_M, theta_U)
