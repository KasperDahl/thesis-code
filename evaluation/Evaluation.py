import pandas as pd


class Evaluation():
    def __init__(self) -> None:
        pass


def prep_precision_recall(data):
    pass


data = pd.read_csv(
    "C:/thesis_code/Github/data/results/junget_1850_1845_EM_scikit")
# data.reset_index
manual_links = pd.read_csv(
    "C:/thesis_code/Github/data/manually_linked/1850_to_1845_full")
print(manual_links)
