from turtle import color
import matplotlib.pyplot as plt


x_labels = ['Junget 1850 to 1845', 'Thy 1850 to 1845', 'Thy 1860 to 1850']
feat_3_after_c = [59.66, 41.55, 15.62]
feat_3_only_man = [54.5, 35.91, 5.58]
feat_5_after_c = [45.54, 99.87, 99.94]
feat_5_only_man = [42.13, 90.1, 34.48]


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
