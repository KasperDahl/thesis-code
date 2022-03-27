import pandas as pd


class Splitset:
    def __init__(self, census_years, year, age):
        self.comp_space = pd.read_csv(
            f"C:/thesis_code/Github/Experiments/data/manual_{census_years}")
        self.full_census = pd.read_csv(f"C:/thesis_code/censuses/LL_data_v1.0.dev2.census/{year}/census.csv", usecols=[
            "pa_id", "sex", "age"])
        self.val_set = pd.read_csv(
            f"C:/thesis_code/Github/data/manually_linked/{census_years}", usecols=['pa_id1', 'pa_id2'])
        self.val_set.drop_duplicates(inplace=True)
        self.census_years = census_years
        self.age = age

        self.val_set = self.val_set.merge(
            self.full_census, how='left', left_on='pa_id1', right_on=['pa_id'])

        self.males_val = self.val_set[self.val_set['sex'] == 'm']
        self.females_val = self.val_set[self.val_set['sex'] == 'f']

        self.above_age = self.val_set[self.val_set['age'] > self.age]
        self.below_age = self.val_set[self.val_set['age'] <= self.age]

        self.comp_males = self.comp_space.merge(
            self.males_val, how='inner', left_on='pa_id_1', right_on='pa_id1')
        self.comp_females = self.comp_space.merge(
            self.females_val, how='inner', left_on='pa_id_1', right_on='pa_id1')

        self.comp_above_age = self.comp_space.merge(
            self.above_age, how='inner', left_on='pa_id_1', right_on='pa_id1')
        self.comp_below_age = self.comp_space.merge(
            self.below_age, how='inner', left_on='pa_id_1', right_on='pa_id1')

        # print(len(self.comp_males), len(self.comp_females), len(
        #     self.comp_males) + len(self.comp_females), len(self.comp_space))

        # print(len(self.comp_above_age), len(self.comp_below_age), len(
        #     self.comp_above_age) + len(self.comp_below_age), len(self.comp_space))

        header = ['fn_score', 'ln_score', 'fam_n_score',
                  'bp_score', 'age_distance', 'pa_id_1', 'pa_id_2']

        self.comp_males.to_csv(
            f"C:/thesis_code/Github/Experiments/data/splitset/{self.census_years}_males", index=False, columns=header)
        self.comp_females.to_csv(
            f"C:/thesis_code/Github/Experiments/data/splitset/{self.census_years}_females", index=False, columns=header)

        self.comp_above_age.to_csv(
            f"C:/thesis_code/Github/Experiments/data/splitset/{self.census_years}_above_age", index=False, columns=header)
        self.comp_below_age.to_csv(
            f"C:/thesis_code/Github/Experiments/data/splitset/{self.census_years}_below_age", index=False, columns=header)


Splitset("1850_1845", "1850", 28)
Splitset("1860_1850", "1860", 33)
