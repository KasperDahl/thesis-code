import matplotlib.pyplot as plt
import numpy as np


labels = ['Junget', 'Thy early',
          'Thy late', 'Val. early',  'Val. late']

pre = [97.3, 96.6, 96.9, 97.4, 97]
recall = [94.5, 92, 93.8, 77.8, 78.6]
fscore = [95.9, 94.2, 95.4, 85, 85.5]


# own_5_pre = [98, 98.8, 99, 99.4, 99.6]
# own_5_rec = [77, 75.9, 64, 76, 76.8]


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
ax.set_title('3 features: Precision/Recall/F-score')
ax.set_xticks([r + barWidth for r in range(len(pre))], labels)
ax.set_ylim([0, 105])
ax.legend(loc='lower right')

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)
ax.bar_label(rects3, padding=3)

fig.tight_layout()
plt.savefig('Experiments/plots/data/bar_plot_pre_rec_own_3_feature.png')
plt.show()
