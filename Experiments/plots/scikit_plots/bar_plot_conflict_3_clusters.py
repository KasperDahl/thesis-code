import matplotlib.pyplot as plt
import numpy as np


labels = ['Junget', 'Thy early',
          'Thy late', 'Val. early',  'Val. late']
sci_3 = [59.66, 41.55, 15.62, 4, 17.2]
sci_5 = [45.54, 99.87, 99.94, 58, 91.3]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars


fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, sci_3, width, label='3 features')
rects2 = ax.bar(x + width/2, sci_5, width, label='5 features')

ax.set_ylabel('Percentage')
ax.set_title('3 cluster scikit-learn EM: set size after conflicts are removed')
ax.set_xticks(x, labels)
ax.set_ylim([0, 108])
ax.legend(loc='upper left')

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)

fig.tight_layout()

plt.savefig('Experiments/plots/data/bar_plot_conflict_3_clusters.png')
plt.show()
