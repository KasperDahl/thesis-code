import matplotlib.pyplot as plt
import numpy as np


labels = ['Junget', 'Thy early',
          'Thy late', 'Val. early',  'Val. late']
own_3 = [97.15, 74.88, 76, 61.5, 61.4]
own_5 = [97.8, 93.12, 93.03, 90.55, 95.1]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars


fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, own_3, width, label='3 features')
rects2 = ax.bar(x + width/2, own_5, width, label='5 features')

ax.set_ylabel('Percentage')
ax.set_title('Set size after conflicts are removed')
ax.set_xticks(x, labels)
ax.set_ylim([0, 108])
ax.legend(loc='lower right')

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)

fig.tight_layout()

plt.savefig('Experiments/plots/data/bar_plot_conflict_own_EM.png')
plt.show()
