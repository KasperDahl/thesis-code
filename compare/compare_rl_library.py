
# initial attempt to compare two df's from the small trainingdatasets
# I will make use of the recordlinkage library = https://github.com/J535D165/recordlinkage

import pandas as pd
import recordlinkage as rl
from recordlinkage.base import BaseCompareFeature

# junget_1845 = pd.read_csv(
#     "C:/thesis_code/Github/data/trainingsets_s/junget_1845")
# junget_1850 = pd.read_csv(
#     "C:/thesis_code/Github/data/trainingsets_s/junget_1850")

thy_1845 = pd.read_csv(
    "C:/thesis_code/Github/data/trainingsets_s/thy_parishes_1845")
thy_1850 = pd.read_csv(
    "C:/thesis_code/Github/data/trainingsets_s/thy_parishes_1850")

## Record Linkage library ##

# I need the age distance therefore the following code - inspired by User-defined algorithm 1:
# https://recordlinkage.readthedocs.io/en/latest/ref-compare.html#user-defined-algorithms


class CompareAge(BaseCompareFeature):

    def _compute_vectorized(self, d1, d2):
        dist = d1 - d2
        return dist


# 1 - Indexing/blocking -
# initialise Index class
indexer = rl.Index()
# indexer.full()
indexer.block(left_on='sex')
candidate_pairs = indexer.index(thy_1850, thy_1845)

# 2 - Comparing step - Jaro-Winkler score for first name and patronym
# initialise Compare class
comp = rl.Compare()
comp.string('first_names', 'first_names',
            method='jaro_winkler', label='fn_score')
comp.string('patronyms', 'patronyms', method='jaro_winkler', label='ln_score')
comp.add(CompareAge('birth_year', 'birth_year', label='age_distance'))

# the method .compute() returns the DataFrame with the feature vectors
comparison_vectors = comp.compute(candidate_pairs, thy_1850, thy_1845)

# changing to absolute values and dropping all age distances above 2
# for optmization purposes this should maybe be done before comparing
comparison_vectors['age_distance'] = comparison_vectors['age_distance'].abs()
comparison_vectors = comparison_vectors.loc[comparison_vectors['age_distance'] <= 2]

comparison_vectors.to_csv(
    "C:/thesis_code/Github/data//comp_sets/thy_parishes_1850_1845")
