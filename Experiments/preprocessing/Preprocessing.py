import pandas as pd
import recordlinkage as rl
from recordlinkage.base import BaseCompareFeature


class Preprocessing:
    def __init__(self, years):
        self.path = "C:/thesis_code/censuses/LL_data_v1.0.dev2.census/"
        self.df = []
        self.years = years
        self.cols = [
            "pa_id",
            "first_names",
            "family_names",
            "patronyms",
            "sex",
            "age",
            "event_parish",
            "birth_parish",
        ]
        self.temp = self.preparation()
        # print(self.temp)

    def preparation(self):
        for year in self.years:
            f = "{}{}/census.csv".format(self.path, year)
            temp = pd.read_csv(
                f, sep=",", usecols=self.cols, index_col="pa_id", dtype="unicode"
            )
            temp["census_year"] = year  # make new column with census year
            self.df.append(temp)
        return pd.concat(self.df)

    def specific_parishes(self, parishes, year):
        df_parishes = []
        for p in parishes:
            df = self.temp.loc[self.temp["event_parish"] == p]
            df['birth_year'] = year - df['age'].astype(float)
            df.reset_index()
            df_parishes.append(df)
        return pd.concat(df_parishes)
        # Apparantly df_parishes is of type list (maybe because of append(), which is also deprecated),
        # therefore I have to use the concat-function to turn it into a dataframe - maybe this can be done more clean


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
    comparison_vectors = comp.compute(candidate_pairs, latest, earliest)

    # changing to absolute values and dropping all age distances above 2
    # for optmization purposes this should maybe be done before comparing
    comparison_vectors['age_distance'] = comparison_vectors['age_distance'].abs()
    comparison_vectors = comparison_vectors.loc[comparison_vectors['age_distance'] <= 2]

    comparison_vectors.to_csv(
        f"C:/thesis_code/Github/Experiments/data/{path}")


latest = Preprocessing(['1850'])
p_latest = latest.specific_parishes(
    ["sevel", "selde", "thorum", "junget"], 1850)
p_latest = latest.specific_parishes(
    ["junget"], 1850)
print(p_latest)
# # p_latest.to_csv("C:/thesis_code/Github/Experiments/data/thy1860")

# earliest = Preprocessing(['1845'])
# # p_earliest = earliest.specific_parishes(
# #     ["sevel", "selde", "thorum", "junget"], 1850)
# p_earliest = earliest.specific_parishes(
#     ["junget"], 1845)
# # print(p_earliest)

# compare = compare(p_latest, p_earliest, "junget_1850_1845")
