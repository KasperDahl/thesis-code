import matplotlib.pyplot as plt
import numpy as np


labels = ['Junget', 'Thy early',
          'Thy late', 'Val. early',  'Val. late']

pre = [83.8, 98, 91.2, 95.6, 99]
recall = [89.2, 50.9, 50.4, 73.9, 76.6]
fscore = [86.2, 51.6, 50.7, 81.4, 84.5]

x = np.arange(len(labels))  # the label locations
barWidth = 0.3  # the width of the bars

r1 = np.arange(len(labels))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

fig, ax = plt.subplots()
rects1 = ax.bar(r1, pre, width=barWidth, label='Precision')
rects2 = ax.bar(r2, recall, width=barWidth, label='Recall')
rects3 = ax.bar(r3, fscore, width=barWidth, label='F-score')


ax.set_ylabel('Percentage')
ax.set_title('3 clusters, 5 features scikit EM: Precision/Recall/F-score')
ax.set_xticks([r + barWidth for r in range(len(pre))], labels)
ax.set_ylim([30, 105])
ax.legend(loc='lower right')

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)
ax.bar_label(rects3, padding=3)
plt.grid(True)
fig.tight_layout()

plt.savefig('Experiments/plots/data/bar_plot_pre_rec_3_cluster_5_feature.png')
# plt.show()
