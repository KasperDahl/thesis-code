from locale import normalize
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def confusion_matrix(years, type):
    df = pd.read_csv(
        f"C:/thesis_code/Github/Experiments/plots/confusion_data/{type}_{years}")
    y_true = df['Correct link'].to_list()
    y_pred = df['Match'].to_list()
    #labels = [0, 1, 2, 3]
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    # plt.savefig(f'Experiments/plots/data/confusion_matrix_{type}_{years}.png')
    plt.show()


confusion_matrix("1850_1845", "baseline")
#confusion_matrix("1850_1845", "splitset_age")
