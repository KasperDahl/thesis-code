import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


data = pd.read_csv(
    "C:/thesis_code/Github/data/results/junget_1850_1845_EM_scikit")
# data.reset_index
# print(data)
manual_links = pd.read_csv(
    "C:/thesis_code/Github/data/manually_linked/1850_to_1845_full")
# print(manual_links)
manual = manual_links['pa_id1'].tolist()


def merge_pa_id(df):
    # merge pa_ids from sources into the dataframe with the EM-results and remove the indices given at comparison level
    # source 1 is the latest census, since the standard is to link backwards
    source_1 = pd.read_csv(
        "C:/thesis_code/Github/data/trainingsets_s/junget_1850", usecols=['pa_id'])
    source_2 = pd.read_csv(
        "C:/thesis_code/Github/data/trainingsets_s/junget_1845", usecols=['pa_id'])
    df1 = df.merge(source_1, left_on='Unnamed: 0', right_index=True)
    df2 = df1.merge(source_2, how='left',
                    left_on='Unnamed: 1', right_index=True, suffixes=('_1850', '_1845'))
    #df2.sort_values(by=['Unnamed: 0'])
    df2.drop(['Unnamed: 0', 'Unnamed: 1'], axis=1, inplace=True)
    # df2.to_csv(
    #     "C:/thesis_code/Github/data/evaluation/junget_1850_1845_pa_id")
    return df2


def remove_conflicts(df):
    # remove all pairs for conflicting matches, meaning the EM links the individual more than once
    grouped = df.groupby('pa_id_1850')
    filtered = grouped.filter(lambda x: (x['EM_results'].sum()) <= 1)
    # might also be possible to do with the code below - consider if need for optimization
    # mask = df.groupby('pa_id_1850')['EM_results'].transform('sum').le(1)
    # df = df[mask]
    return filtered


def remove_non_manual_links(df):
    # remove the links from the comparison space that have not been manually linked
    manual_links = pd.read_csv(
        "C:/thesis_code/Github/data/manually_linked/1850_to_1845_full")
    pa_id1_list = manual_links['pa_id1'].tolist()
    only_manual_links = df.loc[df['pa_id_1850'].isin(pa_id1_list)]
    return only_manual_links


def attach_manual_links(df):
    # attach the pa_id of the correct manual link to the df
    manual_links = pd.read_csv(
        "C:/thesis_code/Github/data/manually_linked/1850_to_1845_full")
    pa_id_1850 = df['pa_id_1850'].tolist()
    only_links = manual_links.loc[manual_links['pa_id1'].isin(
        pa_id_1850)]
    only_links.drop(
        ['Unnamed: 0', 'source_id1', 'source_id2'], axis=1, inplace=True)
    df1 = df.merge(only_links, how='left',
                   left_on='pa_id_1850', right_on='pa_id1')
    df1.drop(
        ['pa_id1'], axis=1, inplace=True)
    # df1.to_csv(
    #     "C:/thesis_code/Github/data/evaluation/junget_1850_1845_pa_id")
    return df1


def find_correct_links(df):
    # create new column with the score 1 for a correct link, and 0 for all incorrect links
    # All the correct manual links that the EM didn't find are not represented - need to be done in some manner
    df['manual_link'] = 0
    counter = 0
    for index, row in df.iterrows():
        if row['pa_id_1845'] == row['pa_id2']:
            row['manual_link'] = 1
            counter += 1
    # print(counter)
    df.to_csv(
        "C:/thesis_code/Github/data/evaluation/junget_1850_1845_pa_id")
    return df


def precision_recall(df):
    results = df['EM_results'].tolist()
    manual_links = df['manual_link'].tolist()
    print(precision_recall_fscore_support(
        manual_links, results, average='macro'))
    # print(results)


# ISSUES:
# 1. No evaluation of links, so in many cases the EM-algorithm links one pa_id to more than one other pa_id

df_pa_id = merge_pa_id(data)
print(df_pa_id)
no_conflicts = remove_conflicts(df_pa_id)
print(no_conflicts)
removed = remove_non_manual_links(no_conflicts)
# # print(removed)
manual = attach_manual_links(removed)
# # print(manual)
correct_links = find_correct_links(manual)
# # print(correct_links)
precision_recall(correct_links)

# find_correct_links(test)
# print(df_pa_id)
# manual_links = pd.read_csv(
#     "C:/thesis_code/Github/data/manually_linked/1850_to_1845_full")
# pa_id1_list = manual_links['pa_id1'].tolist()
# pa_id2_list = manual_links['pa_id2'].tolist()
# df['manual_link'] = 0

# print(df)
# for index, row in df.iterrows():
#     if ((df['pa_id_1850'].isin(pa_id1_list)) & (df['pa_id_1845'].isin(pa_id2_list))):
#         df.set_value(index, 'manual_link', 1)
# print(df)
# correct_links = df.loc[(df['pa_id_1850'].isin(pa_id1_list)) & (
#     df['pa_id_1845'].isin(pa_id2_list))]
# print(correct_links)
