import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from bisect import bisect_left


class Baseline:
    def __init__(self, place, years):
        self.place = place
        self.years = years
        self.comp_space = pd.read_csv(f"C:/thesis_code/Github/Experiments/data/{place}_{years}", usecols=[
                                      'pa_id_1', 'pa_id_2', 'fn_score', 'ln_score', 'age_distance', 'bp_score'])
        self.manual_links = pd.read_csv(
            f"C:/thesis_code/Github/data/manually_linked/{years}", usecols=["pa_id1", "pa_id2"])
        self.data = self.bisect(self.comp_space)
        perfect = self.perfect_scores(self.data)
        self.merged_pa_id = self.comp_space.join(perfect['Match'], how='outer')
        self.merged_pa_id.drop(
            columns=['age_distance', 'fn_score', 'ln_score', 'bp_score'], inplace=True)
        self.match_values = 0
        # below are function from Evaluation
        self.resolved = self.resolve_conflicts(self.merged_pa_id)
        self.only_manual = self.remove_non_manual_link(self.resolved)
        self.attach_manual_id = self.attach_manual_links(self.only_manual)
        self.correct = self.find_correct_links(self.attach_manual_id)

        # variables for size calculations (see string results)
        self.size_original = len(self.comp_space)
        self.size_after_conflicts = len(self.resolved)
        self.size_only_manuals = len(self.only_manual)

        self.precision_recall(self.correct)

    def bisect(self, dataset):
        df = pd.DataFrame(dataset["age_distance"].astype(int))
        df['fn_score'] = pd.Series([self.convert_JW(feature)
                                    for feature in dataset["fn_score"]])
        df['ln_score'] = pd.Series([self.convert_JW(feature)
                                    for feature in dataset["ln_score"]])
        df['bp_score'] = pd.Series([self.convert_JW(feature)
                                    for feature in dataset['bp_score']])
        return df

    def convert_JW(self, feature, breakpoints=[0.75, 0.88, 0.933], values=[3, 2, 1, 0]):
        i = bisect_left(breakpoints, feature)
        return values[i]

    def perfect_scores(self, df):
        conditions = [df['age_distance'].eq(
            0) & df['fn_score'].eq(0) & df['ln_score'].eq(0) & df['bp_score'].eq(0)]
        values = [1]
        df['Match'] = np.select(conditions, values, default=0)
        return df
        # df.to_csv("C:/thesis_code/Github/Experiments/test")

    # def remove_conflicts(self, df):
    #     # remove all pairs for conflicting matches, meaning when the EM links the individual more than once
    #     grouped = df.groupby('pa_id_1')
    #     filtered = grouped.filter(lambda x: (x['Match'].sum()) <= 1)
    #     return filtered

    def resolve_conflicts(self, df):
        # if one individual is matched to more than one individual
        mask_unique = df.groupby(["pa_id_1", "Match"])[
            "Match"].transform(lambda x: len(x) == 1)
        df.loc[:, "Match"] = 1 * (mask_unique)
        return df

    def remove_non_manual_link(self, df):
        # remove the links from the comparison space that have not been manually linked
        pa_id1_list = self.manual_links['pa_id1'].tolist()
        only_manual_links = df.loc[df['pa_id_1'].isin(pa_id1_list)]
        return only_manual_links

    def attach_manual_links(self, df):
        # attach the pa_id of the correct manual link to df
        pa_id_1 = df['pa_id_1'].tolist()
        only_links = self.manual_links.loc[self.manual_links['pa_id1'].isin(
            pa_id_1)]
        df1 = df.merge(only_links.drop_duplicates(subset=['pa_id1']), how='left',
                       left_on='pa_id_1', right_on='pa_id1')
        return df1

    def find_correct_links(self, df):
        conditions = [df['pa_id_2'].eq(df['pa_id2']) & df['Match'].eq(
            1), df['pa_id_2'].ne(df['pa_id2']) & df['Match'].eq(
            1), df['pa_id_2'].eq(df['pa_id2']) & df['Match'].eq(0)]
        values = [1, 2, 3]
        df['Correct link'] = np.select(conditions, values, default=0)
        self.match_values = df['Correct link'].value_counts()
        return df

    def precision_recall(self, df):
        val_0 = len(df[df['Correct link'] == 0])
        val_1 = len(df[df['Correct link'] == 1])
        val_2 = len(df[df['Correct link'] == 2])
        val_3 = len(df[df['Correct link'] == 3])
        df['Correct link'] = np.where(
            df['Correct link'] == 3, 1, df['Correct link'])
        df['Correct link'] = np.where(
            df['Correct link'] == 2, 0, df['Correct link'])
        manual_links = df['Correct link'].tolist()
        results = df['Match'].tolist()
        pre_recall = precision_recall_fscore_support(
            manual_links, results, average='macro')
        f = open(
            f"C:/thesis_code/Github/Experiments/evaluation_results/Baseline_NO_conflicts_4", "a")
        f.write(
            f"\nBaseline 4 features conflicts RESOLVED, place: {self.place}, time: {self.years} \
            \nOriginal size: {self.size_original}, size after conflicts are removed: {self.size_after_conflicts}, percentage of original: {round(((self.size_after_conflicts/self.size_original)*100),2)}%,\
            \nSize after manual links added: {self.size_only_manuals}, percentage of original: {round(((self.size_only_manuals/self.size_original)*100),2)}%,\
            \nMatch values (0 = correct NON-matches, 1 = correct MATCHES, 2 = Type I errors (False positive), 3 = Type II errors (False negative)):    \
            \n{self.match_values}  \
            \nPercentages for table: 0: {(val_0/self.size_only_manuals)*100},  1: {(val_1/self.size_only_manuals)*100},  2: {(val_2/self.size_only_manuals)*100},  3: {(val_3/self.size_only_manuals)*100}   \
            \nPrecision-Recall-score: {pre_recall}\n")


Baseline("junget", "1850_1845")
Baseline("thy_parishes", "1850_1845")
Baseline("thy_parishes", "1860_1850")
Baseline("manual", "1850_1845")
Baseline("manual", "1860_1850")
