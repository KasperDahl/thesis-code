from locale import normalize
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

df = pd.read_csv(
    "C:/thesis_code/Github/Experiments/plots/confusion_data/junget_1850_1845_EM_Abra_3")
y_true = df['Correct link'].to_list()
y_pred = df['Match'].to_list()
#labels = [0, 1, 2, 3]
ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
plt.savefig('Experiments/plots/data/confusion_matrix.png')
# plt.show()


# begge sets skal bare være 0 og 1
