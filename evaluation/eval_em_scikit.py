import pandas as pd

# I need to evaluate the suggested matches based on the following issues (for scikit EM):
# 1. Iterate through the results from EM-algorithm (dataframe) and create new dataframe with only match-cluster
#        ISSUES:
#           - There is no information about the clusters besides the assignment of a number
#           - This means there is no evaluation of the score
#        SOLUTIONS (at this point):
#           - Create more cluster
# 2. Assign the correct pa_id to the "match-cluster" - at this point it is the indices from the comparison step
#         ISSUES:
#           - Currently it is the record_linkage library doing the blocking, this might be different when blocking is perform differently
#         SOLUTIONS:
#           - Is it possible to assign the pa_id instead of the rl_library index numbers to avoid the extra bookkeeping? (Multi-index?) ASK NICOLAI
# 3. Evaluation of links:
#         ISSUES:
#           - One individual might be linked to more than one other individual
#           - Links might be based on a really low score
#          SOLUTIONS:
#           - Assign the feature values to the DF - possibly create a mean value to compare
#           - Decide on a minimum value for a link
#           - In case of link-conflict, choose the link with the highest value (if above the minimum value)
# 4. Compare results to Manual links, using the scikit learn precision/recall score
#


# Comparison of small test set (eg. "junget_1850_1845") to manually linked full set (eg. "1850_to_1845_full")
# Withdraw all unique indices from source_1 (eg. 1850) - maybe it is all of them?
# Find the relevant ids for each of the unique indices in the manually linked set
# - this should show how many of the ids there exist a confirmed manual link for
# Find out which of the unique
#
