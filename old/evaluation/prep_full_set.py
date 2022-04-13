import pandas as pd


cols = ['source_id1', 'source_id2', 'pa_id1', 'pa_id2']
df = pd.read_csv(
    'C:/thesis_code/training_data/LL_data_v1.0.dev2.manual-links/manual_links.csv', usecols=cols)


def prep_evaluation(df, source_1, source_2):
    training_data = df.loc[(df['source_id1'] ==
                            source_1) & (df['source_id2'] == source_2)]
    return training_data


data = prep_evaluation(df, 6, 5)
data.to_csv("C:/thesis_code/Github/data/manually_linked/1860_to_1850_full")
print(data)
