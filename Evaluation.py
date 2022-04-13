import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


class Evaluation:
    def __init__(
        self,
        place,
        years,
        model,
        threshold,
    ):
        self.model = model
        self.place = place
        self.years = years
        self.threshold_initial = threshold
        self.EM_results = pd.read_csv(
            f"C:/thesis_code/Github/Experiments/results_after_EM/{model}/{place}_{years}", usecols=["EM probabilities"])
        self.manual_links = pd.read_csv(
            f"C:/thesis_code/Github/data/manually_linked/{years}", usecols=["pa_id1", "pa_id2"])
        self.comp_space = pd.read_csv(
            f"C:/thesis_code/Github/Experiments/data/{place}_{years}", usecols=["pa_id_1", "pa_id_2"])

        # Create combined df with pa_id's and EM results
        self.merged_pa_id = self.comp_space.join(self.EM_results, how="outer")

        # variables get updated in find_correct_links()-function
        self.match_values = 0

        # function calls
        self.threshold = self.probability_threshold(
            self.merged_pa_id, threshold)
        #self.removed = self.remove_conflicts(self.threshold)
        self.resolved = self.resolve_conflicts(self.threshold)
        self.only_manual = self.remove_non_manual_link(self.resolved)
        self.attach_manual_id = self.attach_manual_links(self.only_manual)
        self.correct = self.find_correct_links(self.attach_manual_id)

        # variables for size calculations (see string results)
        self.size_original = len(self.EM_results)
        self.size_after_conflicts = len(self.resolved)
        self.size_only_manuals = len(self.only_manual)

        self.precision_recall(self.correct)

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
        # Below only used for Qualitative Analysis
        df.to_csv(
            f"C:/thesis_code/Github/Experiments/analysis/data/{self.place}_{self.years}_{self.model}", columns=['pa_id_1', 'pa_id_2', 'Correct link'], index=False)
        return df

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
            f"C:/thesis_code/Github/Experiments/evaluation_results/NO_CONFLICT_{self.model}", "a")
        f.write(
            f"\nModel: {self.model}, place: {self.place}, time: {self.years}, threshold: {self.threshold_initial} \
            \nOriginal size: {self.size_original}, size after conflicts are removed: {self.size_after_conflicts}, percentage of original: {round(((self.size_after_conflicts/self.size_original)*100),2)}%,\
            \nSize after manual links added: {self.size_only_manuals}, percentage of original: {round(((self.size_only_manuals/self.size_original)*100),2)}%,\
            \nMatch values (0 = correct NON-matches, 1 = correct MATCHES, 2 = Type I errors (False positive), 3 = Type II errors (False negative)):    \
            \n{self.match_values}  \
            \nPrecision-Recall-score: {pre_recall}\n")


Evaluation("manual", "1850_1845", "EM_Abra_3", 0.0001)
Evaluation("manual", "1860_1850", "EM_Abra_3", 0.0001)


Evaluation("manual", "1850_1845", "EM_Abra_5", 0.0001)
Evaluation("manual", "1860_1850", "EM_Abra_5", 0.0001)
