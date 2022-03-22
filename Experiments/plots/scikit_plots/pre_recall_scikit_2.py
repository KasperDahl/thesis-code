from turtle import color
import matplotlib.pyplot as plt


x_labels = ['Junget 1850 to 1845', 'Thy 1850 to 1845', 'Thy 1860 to 1850']

sci_3_2_pre = [83, 95, 0]
sci_3_2_recall = [94.5, 93, 0]

sci_5_2_pre = [47.5, 98, 0]
sci_5_2_recall = [48, 51, 0]


plt.ylabel('Percentage')
plt.xlabel('Small test sets')

plt.plot(x_labels, sci_3_2_pre,
         label='3 features: precision', color='green')
plt.plot(x_labels, sci_3_2_recall,
         label='3 features: recall', linestyle="--", color='green')
plt.plot(x_labels, sci_5_2_pre,
         label='5 features: precision', color='purple')
plt.plot(x_labels, sci_5_2_recall,
         label='5 features: recall', linestyle="--", color='purple')

plt.ylim([-5, 105])
plt.legend()
plt.spring()
plt.savefig('Experiments/plots/data/pre_recall_scikit_2_clusters.png')
# plt.show()
