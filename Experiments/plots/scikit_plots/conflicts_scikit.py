from turtle import color
import matplotlib.pyplot as plt


x_labels = ['Junget', 'Thy early',
            'Thy later']
sci_3_2 = [24.17, 8.25, 0]
sci_3_3 = [59.66, 41.55, 15.62]
sci_5_2 = [48.12, 99.87, 0]
sci_5_3 = [45.54, 99.87, 99.94]

plt.ylabel('Percentage')
plt.xlabel('Test sets')

plt.plot(x_labels, sci_3_2,
         label='scikit EM: 3 feat, 2 clusters')
plt.plot(x_labels, sci_3_3,
         label='scikit EM: 3 feat, 3 clusters')
plt.plot(x_labels, sci_5_2,
         label='scikit EM: 5 feat, 2 clusters')
plt.plot(x_labels, sci_5_3,
         label='scikit EM: 3 feat, 3 clusters')


plt.legend()
plt.spring()
plt.show()
