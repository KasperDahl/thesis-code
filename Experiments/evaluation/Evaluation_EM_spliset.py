import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


class Evaluation:
    def __init__(
        self,
        years,
        threshold,
        path
    ):
        self.years = years
        self.path = path
        self.threshold_initial = threshold
        self.EM_results = pd.read_csv(
            f"C:/thesis_code/Github/Experiments/results_after_EM/Splitset/{path}", usecols=["EM probabilities"])
        self.manual_links = pd.read_csv(
            f"C:/thesis_code/Github/data/manually_linked/{years}", usecols=["pa_id1", "pa_id2"])
        self.comp_space = pd.read_csv(
            f"C:/thesis_code/Github/Experiments/data/splitset/{path}", usecols=["pa_id_1", "pa_id_2"])

        # Create combined df with pa_id's and EM results
        self.merged_pa_id = self.comp_space.join(self.EM_results, how="outer")

        # variables get updated in find_correct_links()-function
        self.match_values = 0
        #self.type_I_errors = self.type_II_errors = self.correct_matches = self.correct_non_matches = 0

        # function calls
        self.threshold = self.probability_threshold(
            self.merged_pa_id, threshold)
        self.resolved = self.resolve_conflicts(self.threshold)
        self.only_manual = self.remove_non_manual_link(self.resolved)
        self.attach_manual_id = self.attach_manual_links(self.only_manual)
        self.correct = self.find_correct_links(self.attach_manual_id)

        # variables for size calculations (see string results)
        self.size_original = len(self.EM_results)
        self.size_after_conflicts = len(self.resolved)
        self.size_only_manuals = len(self.only_manual)

        # self.precision_recall(self.correct)

        # temp = self.EM_results["EM probabilities"].value_counts()
        # f = open(
        #     f"C:/thesis_code/Github/Experiments/probability_spread", "a")
        # f.write(
        #     f"\nProbability spread for {self.model} in {self.place} {self.years}\n{temp}\n")

    def probability_threshold(self, df, threshold):
        # Everything above the threshold will be calculated to a match (1), everything behold to a non-match(1)
        df.loc[df['EM probabilities'] >= threshold, 'Match'] = 1
        df['Match'] = df['Match'].fillna(0)
        df['Match'] = df['Match'].astype(int)
        return df

    def resolve_conflicts(self, df):
        # if one individual is matched to more than one individual
        # print(df.head(150))
        max_proba = df.groupby("pa_id_1")["EM probabilities"].transform(
            lambda x: x.eq(x.max()))
        mask_unique = df.groupby(["pa_id_1", "EM probabilities"])[
            "EM probabilities"].transform(lambda x: len(x) == 1)
        df.loc[:, "Match"] = 1 * (max_proba & mask_unique)
        return df

    # def remove_conflicts(self, df):
    #     # remove all pairs for conflicting matches, meaning when the EM links the individual more than once
    #     grouped = df.groupby('pa_id_1')
    #     filtered = grouped.filter(lambda x: (x['Match'].sum()) <= 1)
    #     return filtered

    def remove_non_manual_link(self, df):
        # remove the links from the comparison space that have not been manually linked
        pa_id1_list = self.manual_links['pa_id1'].tolist()
        only_manual_links = df.loc[df['pa_id_1'].isin(pa_id1_list)]
        return only_manual_links

    def attach_manual_links(self, df):
        # attach the pa_id of the correct manual link to df
        # OBS!! There is a weird error for manual links - more
        pa_id_1 = df['pa_id_1'].tolist()
        only_links = self.manual_links.loc[self.manual_links['pa_id1'].isin(
            pa_id_1)]
        # there was a weird error - therefore the drop_duplicates below
        df1 = df.merge(only_links.drop_duplicates(subset=['pa_id1']), how='left',
                       left_on='pa_id_1', right_on='pa_id1')
        return df1

    def find_correct_links(self, df):
        # create new column with the following scores:
        # 1 = EM found correct link
        # 2 = EM assigned link to NON-MATCH link - TYPE I ERROR
        # 3 = EM assigned NON-MATCH to CORRECT link - TYPE II ERROR
        # 0 = EM correct found non-match
        conditions = [df['pa_id_2'].eq(df['pa_id2']) & df['Match'].eq(
            1), df['pa_id_2'].ne(df['pa_id2']) & df['Match'].eq(
            1), df['pa_id_2'].eq(df['pa_id2']) & df['Match'].eq(0)]
        values = [1, 2, 3]
        df['Correct link'] = np.select(conditions, values, default=0)
        self.match_values = df['Correct link'].value_counts()
        # self.correct_non_matches = df['Correct link'].value_counts()[0]
        # self.correct_matches = df['Correct link'].value_counts()[1]
        # self.type_I_errors = df['Correct link'].value_counts()[2]
        # self.type_II_errors = df['Correct link'].value_counts()[3]

        df.to_csv(
            f"C:/thesis_code/Github/Experiments/plots/confusion_data/splitset_age_{self.years}", columns=['Match', 'Correct link'])

        # Below only used for Qualitative Analysis
        # df.to_csv(
        #     f"C:/thesis_code/Github/Experiments/analysis/data/{self.place}_{self.years}_{self.model}", columns=['pa_id_1', 'pa_id_2', 'Correct link'], index=False)

        return df

        # Below only used for confusion matrix:
        # df.to_csv(
        #     f"C:/thesis_code/Github/Experiments/plots/confusion_data/{self.place}_{self.years}_{self.model}", columns=['Match', 'Correct link'])

    def precision_recall(self, df):
        df['Correct link'] = np.where(
            df['Correct link'] == 3, 1, df['Correct link'])
        df['Correct link'] = np.where(
            df['Correct link'] == 2, 0, df['Correct link'])
        manual_links = df['Correct link'].tolist()
        results = df['Match'].tolist()
        pre_recall = precision_recall_fscore_support(
            manual_links, results, average='macro')
        f = open(
            f"C:/thesis_code/Github/Experiments/evaluation_results/splitset_{self.path}", "a")
        f.write(
            f"\nSet: {self.path}, threshold: {self.threshold_initial} \
            \nOriginal size: {self.size_original}, size after conflicts are removed: {self.size_after_conflicts}, percentage of original: {round(((self.size_after_conflicts/self.size_original)*100),2)}%,\
            \nSize after manual links added: {self.size_only_manuals}, percentage of original: {round(((self.size_only_manuals/self.size_original)*100),2)}%,\
            \nMatch values (0 = correct NON-matches, 1 = correct MATCHES, 2 = Type I errors (False positive), 3 = Type II errors (False negative)):    \
            \n{self.match_values}  \
            \nPrecision-Recall-score: {pre_recall}\n")


#Evaluation("1860_1850", 0.0001, "1860_1850_gender")
Evaluation("1850_1845", 0.0001, "1850_1845_age")
Evaluation("1860_1850", 0.0001, "1860_1850_age")
