import matplotlib.pyplot as plt
import numpy as np


labels = ['Junget', 'Thy early',
          'Thy late', 'Val. early',  'Val. late']

pre_conf = [98.7, 96.9, 94.7, 96.1, 94.3]
pre_no_conf = [98.4, 96.9, 94.7, 96.1, 94.3]


x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, pre_conf, width, label='Conflicts unresolved')
rects2 = ax.bar(x + width/2, pre_no_conf, width, label='Conflicts resolved')


ax.set_ylabel('Percentage')
ax.set_title(
    'Baseline 3 features: Precision')
ax.set_xticks(x, labels)
ax.set_ylim([30, 105])
ax.legend(loc='lower right')

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)

fig.tight_layout()
plt.savefig(
    'Experiments/plots/data/bar_baseline_precision.png')
plt.show()
