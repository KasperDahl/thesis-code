import matplotlib.pyplot as plt
import numpy as np


labels = ['Junget', 'Thy early',
          'Thy late', 'Val. early',  'Val. late']

pre = [98.3, 98.8, 99, 99.4, 99.6]
recall = [77.2, 75.9, 64, 76.5, 77.4]
fscore = [88.3, 83.7, 71.7, 84.5, 85.3]


x = np.arange(len(labels))  # the label locations
barWidth = 0.28  # the width of the bars

r1 = np.arange(len(labels))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

fig, ax = plt.subplots()
rects1 = ax.bar(r1, pre, width=barWidth, label='Precision')
rects2 = ax.bar(r2, recall, width=barWidth, label='Recall')
rects3 = ax.bar(r3, fscore, width=barWidth, label='F-score')


ax.set_ylabel('Percentage')
ax.set_title('5 features: Precision/Recall/F-score')
ax.set_xticks([r + barWidth for r in range(len(pre))], labels)
ax.set_ylim([40, 103])
ax.legend(loc='lower right')

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)
ax.bar_label(rects3, padding=3)

plt.grid(True)
fig.tight_layout()
plt.savefig('Experiments/plots/data/bar_plot_pre_rec_own_5_feature.png')
plt.show()
