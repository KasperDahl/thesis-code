import matplotlib.pyplot as plt
import numpy as np


labels = ['Precision',  'Recall', 'F-score']

baseline = [94.3, 69, 76.6]
abra = [90.2, 81.5, 85.4]
own = [99.4, 78, 85.7]


x = np.arange(len(labels))  # the label locations
width = 0.28  # the width of the bars

r1 = np.arange(len(labels))
r2 = [x + width for x in r1]
r3 = [x + width for x in r2]

fig, ax = plt.subplots()
rects1 = ax.bar(r1, baseline, width, label='Baseline')
rects2 = ax.bar(r2, abra, width, label='Abramitzsky baseline')
rects3 = ax.bar(r3, own, width, label='Best EM model')


ax.set_ylabel('Percentage', fontsize=18)
ax.set_title('Models compared for Validation set 1860 to 1850')
ax.set_xticks([r + width for r in range(len(baseline))], labels)
ax.set_ylim([40, 105])
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)
ax.bar_label(rects3, padding=3)

plt.grid(True)
fig.tight_layout()
plt.savefig(
    'Experiments/plots/data/bar_5_EM_All_1860_1850.png')
plt.show()
