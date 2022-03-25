import matplotlib.pyplot as plt
import numpy as np


labels = ['Junget', 'Thy early',
          'Thy late', 'Val. early',  'Val. late']

pre = [83, 95.3, 0, 91.6, 0]
recall = [94.4, 93, 0, 78.3, 0]
fscore = [87.7, 94.2, 0, 83.7, 0]

# sci_5_2_pre = [47.5, 98, 0, 25, 88.7]
# sci_5_2_recall = [48, 51, 0, 25, 61.2]
# sci_5_2_fscore = [47.8, 51.6, 0, 25, 67.4]

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
ax.set_title('2 clusters, 3 features scikit EM: Precision/Recall/F-score')
ax.set_xticks([r + barWidth for r in range(len(pre))], labels)
ax.set_ylim([0, 105])
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)
ax.bar_label(rects3, padding=3)

fig.tight_layout()
plt.savefig('Experiments/plots/data/bar_plot_pre_rec_2_cluster_3_feature.png')
plt.show()
