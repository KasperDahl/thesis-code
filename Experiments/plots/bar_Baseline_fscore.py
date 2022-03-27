import matplotlib.pyplot as plt
import numpy as np


labels = ['Junget', 'Thy early',
          'Thy late', 'Val. early',  'Val. late']

fscore_conf = [92.9, 88, 84.8, 80, 78.4]
fscore_no_conf = [93, 86.8, 83.2, 77.1, 76.6]


x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, fscore_conf, width, label='Conflicts unresolved')
rects2 = ax.bar(x + width/2, fscore_no_conf, width, label='Conflicts resolved')


ax.set_ylabel('Percentage')
ax.set_title(
    'Baseline 3 features: F-score')
ax.set_xticks(x, labels)
ax.set_ylim([30, 105])
ax.legend(loc='lower right')

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)

fig.tight_layout()

plt.savefig(
    'Experiments/plots/data/bar_baseline_fscore_.png')
plt.show()
