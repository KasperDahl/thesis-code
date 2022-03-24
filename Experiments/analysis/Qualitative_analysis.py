import pandas as pd

# I need to load df with evaluation results
# I will load the df with the "Correct link"-scores of 0, 1, 2 or 3, where:
# 1 = EM found correct link
# 2 = EM assigned link to NON-MATCH link - TYPE I ERROR
# 3 = EM assigned NON-MATCH to CORRECT link - TYPE II ERROR
# 0 = EM correct found non-match

# Then I decide what I want to analyse, initially it should probably be Type II errors, denoted by 3
# I isolate this group and fetch the relevant columns from the original census
# relevant columns: pa_id gender, first name, patronym, family name, birth parish, event county and event parish
# Initially I will perform value count on gender, event county and event parish


def analysis(place, census_years, model, year):
    df = pd.read_csv(
        f"C:/thesis_code/Github/Experiments/analysis/data/{place}_{census_years}_{model}")
    org_df = pd.read_csv(f"C:/thesis_code/censuses/LL_data_v1.0.dev2.census/{year}/census.csv", usecols=[
                         "pa_id", "first_names", "family_names", "patronyms", "sex", "age", "event_parish", "event_county", "birth_parish"])
    only_typeII = df[df['Correct link'] == 3]
    length = len(only_typeII)
    columns = only_typeII.merge(org_df, how='left', left_on=[
                                'pa_id_1'], right_on=['pa_id'])
    gender = columns['sex'].value_counts()
    event_parish = columns['event_parish'].value_counts()
    event_county = columns['event_county'].value_counts()
    age = columns['age'].value_counts().head(20)
    f = open(
        f"C:/thesis_code/Github/Experiments/analysis/results/{model}_{census_years}_no conflicts", "a")
    f.write(f"\n{census_years} TYPE II errors (length: {length}) \nGender spread:\n{gender}\nAge:\n{age}\nEvent parish:\n{event_parish}\nEvent county:\n{event_county}\n")
    # print(columns['sex'].value_counts())


analysis('manual', '1850_1845', 'EM_Abra_3', "1850")
analysis('manual', '1860_1850', 'EM_Abra_3', "1860")

analysis('manual', '1850_1845', 'EM_Abra_5', "1850")
analysis('manual', '1860_1850', 'EM_Abra_5', "1860")
