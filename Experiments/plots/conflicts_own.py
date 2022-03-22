from turtle import color
import matplotlib.pyplot as plt


x_labels = ['Junget', 'Thy early',
            'Thy later', "Val. set early", "Val. set later"]
own_3_after_c = [97.15, 74.88, 76, 61.5, 61.4]
own_5_after_c = [97.8, 93.12, 93.03, 90.55, 95.1]

plt.ylabel('Percentage')
plt.xlabel('Test sets')

plt.plot(x_labels, own_3_after_c,
         label='3 features: Conflicts removed', color='red')
plt.plot(x_labels, own_5_after_c,
         label='5 features: Conflicts removed', color='blue')


plt.ylim([50, 105])
plt.legend()
plt.spring()
plt.savefig('Experiments/plots/data/conflicts_own.png')
# plt.show()
