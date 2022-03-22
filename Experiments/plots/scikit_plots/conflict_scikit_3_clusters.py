from turtle import color
import matplotlib.pyplot as plt


x_labels = ['Junget 1850 to 1845', 'Thy 1850 to 1845', 'Thy 1860 to 1850']
sci_3 = [59.66, 41.55, 15.62]
sci_5 = [45.54, 99.87, 99.94]

plt.ylabel('Percentage')
plt.xlabel('Small test sets')

plt.plot(x_labels, sci_3,
         label='3 features: conflicts removed', color='green')
plt.plot(x_labels, sci_5,
         label='5 features: conflicts removed', color='purple')

plt.ylim([-5, 105])
plt.legend()
plt.spring()
plt.savefig('Experiments/plots/data/conflicts_scikit_3_clusters.png')
# plt.show()
