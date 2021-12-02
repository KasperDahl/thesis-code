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

# 2. Parameters

# 3-dimensional array consisting of the multinomial distance bins (fn, ln, age)
# LOOK INTO: how to automate this (could be min-max range calculation - otherwise just received as parameters)
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
# CH - Looks good enough for an initial value. If you want p_U in a separate variable, you should probably
# CH - calculate it inside the loop so it gets updated whenever p_M changes.
# KDA - true, I will look into that in the EM-steps

# 3. Loop over steps E and M


def em_steps():

    # initialize loop - number of iterations will be changed later
    # loop until convergence - or maximum 100 iterations (or other number)
    # if the latter - this information should be outputted
    n_iterations = 3
    for i in range(n_iterations):
        # E-step

        # CH - Note: Here you need to iterate over your training examples somehow.
        # CH - For each example, you need to (a) get the features and (b) look up the corresponding
        # CH - elements in the theta vector.

        # get features and look up corresponding theta element
        # I wonder if there is a cleaner way to do this
        for row in np.nditer(dataset_values):
            index = dataset_values[row]
            theta_value_M = theta_M[index[0], index[1], index[2]]
            theta_value_U = theta_U[index[0], index[1], index[2]]
            # vector w
            w = (theta_value_M*p_M)/((theta_value_M*p_M)+(theta_value_U*p_U))
        print(f"{i} iteration done: {w} and {theta_value_M}")
        # M-step
        # updated match probability
        p_M_1 = np.mean(w)
        # this is obviously wrong - I have not introduced the the actual

        # updated parameter estimates(theta)
        theta_M_1 = np.argmax(np.sum(np.log(w)))


print(f"starting loop")
em_steps()
