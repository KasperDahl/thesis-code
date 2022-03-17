import pandas as pd

comp = pd.read_csv(
    "C:/thesis_code/Github/Experiments/data/manual_1850_1845_first")
latest = pd.read_csv(
    "C:/thesis_code/Github/Experiments/data/manual_1850_1845_pa_id/1850", usecols=['pa_id1'])
earliest = pd.read_csv(
    "C:/thesis_code/Github/Experiments/data/manual_1850_1845_pa_id/1845", usecols=['pa_id2'])


def merge_pa_id(df, latest, earliest):
    # merge pa_ids from sources into the dataframe with the EM-results and remove the indices given at comparison level
    # source 1 is the latest census, since the standard is to link backwards
    df1 = df.merge(latest, left_on='Unnamed: 0', right_index=True)
    df2 = df1.merge(earliest, how='left',
                    left_on='Unnamed: 1', right_index=True, suffixes=('_1850', '_1845'))
    #df2.sort_values(by=['Unnamed: 0'])
    df2.drop(['Unnamed: 0', 'Unnamed: 1'], axis=1, inplace=True)
    # df2.rename(columns={"pa_id1": "pa_id_1",
    #                     "pa_id2": "pa_id_2"}, inplace=True)
    df2.to_csv(
        "C:/thesis_code/Github/Experiments/data/manual_1850_1845", index=False)
    return df2


res = merge_pa_id(comp, latest, earliest)
print(res)
