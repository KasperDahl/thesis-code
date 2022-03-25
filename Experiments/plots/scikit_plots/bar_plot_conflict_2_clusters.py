import matplotlib.pyplot as plt
import numpy as np


labels = ['Junget', 'Thy early',
          'Thy late', 'Val. early',  'Val. late']
sci_3 = [24.17, 8.25, 0, 1.85, 0]
sci_5 = [48.1, 99.8, 0, 0, 28.88]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars


fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, sci_3, width, label='3 features')
rects2 = ax.bar(x + width/2, sci_5, width, label='5 features')

ax.set_ylabel('Percentage')
ax.set_title('2 cluster scikit-learn EM: set size after conflicts are removed')
ax.set_xticks(x, labels)
ax.set_ylim([0, 108])
ax.legend(loc='upper left')

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)

fig.tight_layout()

plt.savefig('Experiments/plots/data/bar_plot_conflict_2_clusters.png')
plt.show()
