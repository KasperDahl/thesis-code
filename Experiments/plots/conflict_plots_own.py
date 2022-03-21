from turtle import color
import matplotlib.pyplot as plt


x_labels = ['Junget 1850 to 1845', 'Thy 1850 to 1845', 'Thy 1860 to 1850']
own_3_after_c = [97.15, 75.88, 76]
own_3_only_man = [89.53, 67.25, 25.62]
own_5_after_c = [97.8, 93.12, 93.03]
own_5_only_man = [90.17, 84.08, 33.12]


plt.ylabel('Percentage')
plt.xlabel('Small test sets')

plt.plot(x_labels, own_3_after_c,
         label='3 features: conflicts removed', color='blue')
plt.plot(x_labels, own_3_only_man,
         label='3 features: only validation pairs', linestyle="--", color='blue')
plt.plot(x_labels, own_5_after_c,
         label='5 features: conflicts removed', color='red')
plt.plot(x_labels, own_5_only_man,
         label='5 features: only validation pairs', linestyle="--", color='red')
plt.legend()
plt.spring()
plt.show()
