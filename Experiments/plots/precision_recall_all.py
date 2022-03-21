from turtle import color
import matplotlib.pyplot as plt


x_labels = ['Junget 1850 to 1845', 'Thy 1850 to 1845', 'Thy 1860 to 1850']
own_3_pre = [0.9737, 0.9667, 0.9697]
own_3_recall = [0.9451, 0.9206, 0.9387]
own_5_pre = [0.9829, 0.9881, 0.9902]
own_5_recall = [0.7719, 0.7590, 0.6407]
# sci_3_2_pre = []
# sci_3_2_recall = []
# sci_5_2_pre = []
# sci_5_2_recall = []
# sci_3_3_pre = []
# sci_3_3_recall = []
# sci_5_3_pre = []
# sci_5_3_recall = []

plt.ylabel('Percentage')
plt.xlabel('Small test sets')

plt.plot(x_labels, own_3_pre,
         label='3 features: precision', color='blue')
plt.plot(x_labels, own_3_recall,
         label='3 features: recall', linestyle="--", color='blue')
plt.plot(x_labels, own_5_pre,
         label='5 features: precision', color='red')
plt.plot(x_labels, own_5_recall,
         label='5 features: recall', linestyle="--", color='red')
plt.legend()
plt.spring()
plt.show()
