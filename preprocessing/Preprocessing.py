import pandas as pd


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
        ]

    def preparation(self):
        for year in self.years:
            f = "{}{}/census.csv".format(self.path, year)
            temp = pd.read_csv(
                f, sep=",", usecols=self.cols, index_col="pa_id", dtype="unicode"
            )
            temp["census_year"] = year  # make new column with census year
            self.df.append(temp)
        return pd.concat(self.df)

    def specific_parishes(self, temp, parishes, year):
        df_parishes = []
        for p in parishes:
            df = temp.loc[temp["event_parish"] == p]
            df['birth_year'] = year - df['age'].astype(float)
            df.reset_index()
            df_parishes.append(df)
        return pd.concat(df_parishes)
        # Apparantly df_parishes is of type list (maybe because of append(), which is also deprecated),
        # therefore I have to use the concat-function to turn it into a dataframe - maybe this can be done more clean


pre_1850 = Preprocessing(["1850"])
full_1850 = pre_1850.preparation()
pre_1850.specific_parishes(
    full_1850, ["sevel", "selde", "thorum", "junget"], 1850).to_csv("C:/thesis_code/Github/data/trainingsets_s/thy_parishes_1850")
# parishes_1850 = pre_1850.specific_parishes(
#     full_1850, ["sevel", "selde", "thorum", "junget"], 1850)

pre_1845 = Preprocessing(["1845"])
full_1845 = pre_1845.preparation()
pre_1845.specific_parishes(
    full_1845, ["sevel", "selde", "thorum", "junget"], 1845).to_csv("C:/thesis_code/Github/data/trainingsets_s/thy_parishes_1845")
