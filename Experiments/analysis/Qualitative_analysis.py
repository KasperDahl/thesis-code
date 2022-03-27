import pandas as pd


class Analysis:
    def __init__(self, census_years, year, age):
        self.df = pd.read_csv(
            f"C:/thesis_code/Github/Experiments/analysis/data/manual_{census_years}_EM_Abra_5")
        self.org_df = pd.read_csv(f"C:/thesis_code/censuses/LL_data_v1.0.dev2.census/{year}/census.csv", usecols=[
            "pa_id", "first_names", "family_names", "patronyms", "sex", "age", "event_parish", "event_county", "birth_parish"])
        self.manual_df = pd.read_csv(
            f"C:/thesis_code/Github/data/manually_linked/{census_years}", usecols=['pa_id1', 'pa_id2'])
        self.manual_df.drop_duplicates(inplace=True)

        self.age = age
        self.census_years = census_years

        self.length_manual = len(self.manual_df)
        self.length_typeII = 0

        self.manual_res = self.manual_results(self.age)
        self.typeII_res = self.typeII_results(self.age)
        self.result_txt(self.manual_res, self.typeII_res)

    def manual_results(self, age):
        df = self.manual_df.merge(
            self.org_df, how='left', left_on='pa_id1', right_on=['pa_id'])
        amount_men = df.sex.value_counts().m
        amount_women = df.sex.value_counts().f
        above = (df['age'] > age).sum()
        below = (df['age'] <= age).sum()
        #no_age = df['age'].isna().sum()
        return [amount_men, amount_women, above, below]

    def typeII_results(self, age):
        only_typeII = self.df[self.df['Correct link'] == 3]
        self.length_typeII = len(only_typeII)
        df = only_typeII.merge(self.org_df, how='left', left_on=[
            'pa_id_1'], right_on=['pa_id'])
        amount_men = df.sex.value_counts().m
        amount_women = df.sex.value_counts().f
        above = (df['age'] > age).sum()
        below = (df['age'] <= age).sum()
        #no_age = df['age'].isna().sum()
        return [amount_men, amount_women, above, below]

    def result_txt(self, manual_res, typeII_res):
        f = open(
            f"C:/thesis_code/Github/Experiments/analysis/results/Abra_5_manual_{self.census_years}.txt", "a")
        # f.write(f"\
        # \nValidation set: {self.census_years}, set length: {self.length_manual}, TYPE II errors length: {self.length_typeII} \
        # \nGender in Val. set: \nMen: {manual_res[0]}, Percentage: {(manual_res[0]/self.length_manual)*100}  \
        # \nWomen: {manual_res[1]}, Percentage: {(manual_res[1]/self.length_manual)*100}  \
        # \nGender in Type II set: \nMen: {typeII_res[0]}, Percentage: {(typeII_res[0]/self.length_typeII)*100}  \
        # \nWomen: {typeII_res[1]}, Percentage: {(typeII_res[1]/self.length_typeII)*100}  \
        # \nTHRESHOLD AGE: {self.age}\
        # \nAge in Val. set:  \nAbove: {manual_res[2]}, Percentage above: {(manual_res[2]/self.length_manual)*100}  \
        # \nBelow: {manual_res[3]}, Percentage below: {(manual_res[3]/self.length_manual)*100}  \
        # \nAge in Type II set: \nAbove: {typeII_res[2]}, Percentage: {(typeII_res[2]/self.length_typeII)*100}  \
        # \nBelow: {typeII_res[3]}, Percentage below: {(typeII_res[3]/self.length_typeII)*100}  \
        # \n")
        f.write(f"\
        \nTHRESHOLD AGE: {self.age}\
        \nAge in Val. set:  \nAbove: {manual_res[2]}, Percentage above: {(manual_res[2]/self.length_manual)*100}  \
        \nBelow: {manual_res[3]}, Percentage below: {(manual_res[3]/self.length_manual)*100}  \
        \nAge in Type II set: \nAbove: {typeII_res[2]}, Percentage: {(typeII_res[2]/self.length_typeII)*100}  \
        \nBelow: {typeII_res[3]}, Percentage below: {(typeII_res[3]/self.length_typeII)*100}  \
        \n")


# Analysis('1860_1850', "1860", 13)
# Analysis('1850_1845', "1850", 13)
# Analysis('1860_1850', "1860", 18)
# Analysis('1850_1845', "1850", 18)

# Analysis('1860_1850', "1860", 23)
# Analysis('1850_1845', "1850", 23)
# Analysis('1860_1850', "1860", 28)
# Analysis('1850_1845', "1850", 28)
Analysis('1860_1850', "1860", 33)
