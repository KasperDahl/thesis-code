from turtle import color
import matplotlib.pyplot as plt


x_labels = ['Junget 1850 to 1845', 'Thy 1850 to 1845', 'Thy 1860 to 1850']
feat_3_after_c = [24.17, 8.25, 0]
feat_3_only_man = [21.18, 6.88, 0]
feat_5_after_c = [48.12, 99.87, 0]
feat_5_only_man = [45.5, 90.1, 0]


plt.ylabel('Percentage')
plt.xlabel('Small test sets')

plt.plot(x_labels, feat_3_after_c,
         label='3 features: conflicts removed', color='blue')
plt.plot(x_labels, feat_3_only_man,
         label='3 features: only validation pairs', linestyle="--", color='blue')
plt.plot(x_labels, feat_5_after_c,
         label='5 features: conflicts removed', color='red')
plt.plot(x_labels, feat_5_only_man,
         label='5 features: only validation pairs', linestyle="--", color='red')
plt.legend()
plt.spring()
plt.show()
