import pandas as pd
import numpy as np


dataset = pd.read_csv('features.csv')
fn_feature = dataset['fn_score']
ln_feature = dataset['ln_score']
dist_age = dataset['age_distance']

# 1. Distance Bins

# The string numbers below are based on research by Winkler 1988
# OBS - I am guessing there is a cleverer way to do this in Python - will look into it later on
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


# Create numpy array with values for use in E-step (maybe there is cleaner way to do this)
df = pd.DataFrame({'age': dist_age, 'fn': dist_fn, 'ln': dist_ln})
dataset_values = df[['age', 'fn', 'ln']].to_numpy()

#dataset_values = np.array([dist_age, dist_fn, dist_ln])

# 2. Parameters

# 3-dimensional arrays of dimension (n_fn, n_ln, n_age)
# CH - Fine. Perhaps it's better to define the thetas as 3-dimensional arrays of dimension (n_fn, n_ln, n_age) though.
# KDA: Is this what you meant?
dimensions = len(dist_age)*len(dist_fn)*len(dist_ln)
theta_M, theta_U = np.full((len(dist_age), len(dist_fn), len(
    dist_ln)), 1/dimensions)

# probability of being a match - should be a reasonable guess on my part on amount of matches
# if I estimate a 100 matches in the comparison space of two censuses of a 1000 records each
# the estimated probability is 100/(1000 X 1000) = 0,0001
p_M = 0.0001
# or: p_M = match_guess/(len(census_1)*len(census_2))
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
        for row in range(dataset_values):
            index = dataset_values[row]
            theta_value_M = theta_M[index[0], index[1], index[2]]
            theta_value_U = theta_U[index[0], index[1], index[2]]
            # vector w
            w = (theta_value_M*p_M)/((theta_value_M*p_M)+(theta_value_U*p_U))

        # M-step
        # updated match probability
        p_M_1 = np.mean(w)
        # this is obviously wrong - I have not introduced the the actual

        # updated parameter estimates(theta)
        theta_M_1 = np.argmax(np.sum(np.log(w)))
