from turtle import color
import matplotlib.pyplot as plt


x_labels = ['Junget 1850 to 1845', 'Thy 1850 to 1845', 'Thy 1860 to 1850']

sci_3_3_pre = [91.7, 93.8, 87.5]
sci_3_3_recall = [96.7, 93.5, 81.7]

sci_5_3_pre = [83.8, 98, 91]
sci_5_3_recall = [89.2, 50.9, 50.5]

plt.ylabel('Percentage')
plt.xlabel('Small test sets')

plt.plot(x_labels, sci_3_3_pre,
         label='3 features: precision', color='green')
plt.plot(x_labels, sci_3_3_recall,
         label='3 features: recall', linestyle="--", color='green')
plt.plot(x_labels, sci_5_3_pre,
         label='5 features: precision', color='purple')
plt.plot(x_labels, sci_5_3_recall,
         label='5 features: recall', linestyle="--", color='purple')

plt.ylim([-5, 105])
plt.legend()
plt.spring()
plt.savefig('Experiments/plots/data/pre_recall_scikit_3_clusters.png')
# plt.show()
