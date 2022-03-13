import pandas as pd

## following code is from Nicolai Rask ##

path = "C:/thesis_code/censuses/LL_data_v1.0.dev2.census/"
df = []  # container for census data
# census years to load
years = [1850]
# columns to read
cols = ["pa_id", "first_names", "family_names", "patronyms",
        "sex", "age", "event_parish", "birth_parish"]
for year in years:
    f = "{}{}/census.csv".format(path, year)
    temp = pd.read_csv(f, sep=",", usecols=cols,
                       index_col="pa_id", dtype="unicode")
    temp["census_year"] = year  # make new column with census year
    df.append(temp)

census_1850 = pd.concat(df)  # concatenate into one single DataFrame
# census_1850.to_csv("C:/thesis_code/Github/data/censuses/census_1850")
print(census_1850)
