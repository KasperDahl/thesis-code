import pandas as pd
import recordlinkage as rl
from recordlinkage.base import BaseCompareFeature


class CompareAge(BaseCompareFeature):

    def _compute_vectorized(self, d1, d2):
        dist = d1 - d2
        return dist


def compare(latest, earliest, path):
    # Blocking
    indexer = rl.Index()
    indexer.block(left_on='sex')
    candidate_pairs = indexer.index(latest, earliest)
    # Comparing step
    comp = rl.Compare()
    comp.string('first_names', 'first_names',
                method='jaro_winkler', label='fn_score')
    comp.string('patronyms', 'patronyms',
                method='jaro_winkler', label='ln_score')
    comp.string('family_names', 'family_names',
                method='jaro_winkler', label='fam_n_score')
    comp.string('birth_parish', 'birth_parish',
                method='jaro_winkler', label='bp_score')
    comp.add(CompareAge('birth_year', 'birth_year', label='age_distance'))
    print("start comp_vec")
    comparison_vectors = comp.compute(candidate_pairs, latest, earliest)
    print("end comp_vec")
    # changing to absolute values and dropping all age distances above 2
    # for optmization purposes this should maybe be done before comparing
    comparison_vectors['age_distance'] = comparison_vectors['age_distance'].abs()
    comparison_vectors = comparison_vectors.loc[comparison_vectors['age_distance'] <= 2]

    comparison_vectors.to_csv(
        f"C:/thesis_code/Github/Experiments/data/manual_{path}")


cols = ["pa_id", "first_names", "family_names", "patronyms",
        "sex", "age", "event_parish", "birth_parish"]

# manual_1850_1845 = pd.read_csv(
#     "C:/thesis_code/Github/data/manually_linked/1850_1845", usecols=['pa_id1', 'pa_id2'])

# full_1850 = pd.read_csv(
#     "C:/thesis_code/censuses/LL_data_v1.0.dev2.census/1850/census.csv", usecols=cols)
# full_1845 = pd.read_csv(
#     "C:/thesis_code/censuses/LL_data_v1.0.dev2.census/1845/census.csv", usecols=cols)

# manual_1850 = manual_1850_1845.merge(
#     full_1850, how='inner', left_on='pa_id1', right_on='pa_id')
# manual_1850.drop(['pa_id', 'pa_id2'], axis=1, inplace=True)
# manual_1850['birth_year'] = 1850 - manual_1850['age'].astype(float)
# manual_1850.to_csv(
#     "C:/thesis_code/Github/Experiments/data/manual_1850_1845_pa_id/1850")

# manual_1845 = manual_1850_1845.merge(
#     full_1845, how='inner', left_on='pa_id2', right_on='pa_id')
# manual_1845.drop(['pa_id1', 'pa_id'], axis=1, inplace=True)
# manual_1845['birth_year'] = 1845 - manual_1845['age'].astype(float)
# manual_1845.to_csv(
#     "C:/thesis_code/Github/Experiments/data/manual_1850_1845_pa_id/1845")

# compare = compare(manual_1850, manual_1845, "1850_1845")

manual_1860_1850 = pd.read_csv(
    "C:/thesis_code/Github/data/manually_linked/1860_1850", usecols=['pa_id1', 'pa_id2'])

full_1860 = pd.read_csv(
    "C:/thesis_code/censuses/LL_data_v1.0.dev2.census/1860/census.csv", usecols=cols)
full_1850 = pd.read_csv(
    "C:/thesis_code/censuses/LL_data_v1.0.dev2.census/1850/census.csv", usecols=cols)

manual_1860 = manual_1860_1850.merge(
    full_1860, how='inner', left_on='pa_id1', right_on='pa_id')
manual_1860.drop(['pa_id', 'pa_id2'], axis=1, inplace=True)
manual_1860['birth_year'] = 1860 - manual_1860['age'].astype(float)
manual_1860.to_csv(
    "C:/thesis_code/Github/Experiments/data/manual_1860_1850_pa_id/1860")

manual_1850 = manual_1860_1850.merge(
    full_1850, how='inner', left_on='pa_id2', right_on='pa_id')
manual_1850.drop(['pa_id1', 'pa_id'], axis=1, inplace=True)
manual_1850['birth_year'] = 1850 - manual_1850['age'].astype(float)
manual_1850.to_csv(
    "C:/thesis_code/Github/Experiments/data/manual_1860_1850_pa_id/1850")

compare = compare(manual_1860, manual_1850, "1860_1850")
