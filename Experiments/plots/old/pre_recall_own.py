from turtle import color
import matplotlib.pyplot as plt


x_labels = ['Junget', 'Thy early',
            'Thy later', "Val. set early", "Val. set later"]

own_3_pre = [97, 96.6, 96.9, 97.4, 97]
own_3_rec = [94.5, 92, 93.8, 77.5, 79]

own_5_pre = [98, 98.8, 99, 99.4, 99.6]
own_5_rec = [77, 75.9, 64, 76, 76.8]

# own_3_after_c = [97.15, 74.88, 76, 61.5, 61.4]
# own_5_after_c = [97.8, 93.12, 93.03, 90.55, 95.1]

plt.ylabel('Percentage')
plt.xlabel('Test sets')

plt.plot(x_labels, own_3_pre,
         label='3 features: Precision score', color='red')
plt.plot(x_labels, own_3_rec,
         label='3 features: Recall score', color='red', linestyle='--')
plt.plot(x_labels, own_5_pre,
         label='5 features: Precision score', color='blue')
plt.plot(x_labels, own_5_rec,
         label='5 features: Recall score', color='blue', linestyle='--')

# plt.plot(x_labels, own_3_after_c,
#          label='3 features: Size after conflicts', color='red', linestyle='-.')
# plt.plot(x_labels, own_5_after_c,
#          label='5 features: Size after conflicts', color='red', linestyle='-.')

plt.ylim([50, 105])
plt.legend()
plt.spring()
# plt.show()
plt.savefig('Experiments/plots/data/pre_recall_own.png')
