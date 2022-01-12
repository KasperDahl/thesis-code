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
