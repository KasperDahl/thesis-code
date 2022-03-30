import matplotlib.pyplot as plt
import numpy as np


labels = ['Precision',  'Recall', 'F-score']

baseline = [94.3, 69, 76.6]
own = [99.4, 78, 85.7]


x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, baseline, width, label='Baseline')
rects2 = ax.bar(x + width/2, own, width, label='Best EM model')


ax.set_ylabel('Percentage', fontsize=18)
#ax.set_title('Conflicts resolved: Val. set  1850 to 1845')
ax.set_xticks(x, labels, fontsize=18)
ax.set_ylim([40, 105])
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)

plt.grid(True)
fig.tight_layout()
plt.savefig(
    'Experiments/plots/data/bar_5_EM_baseline_1860_1850.png')
plt.show()
