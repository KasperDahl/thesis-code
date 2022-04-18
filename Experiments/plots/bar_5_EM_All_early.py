import matplotlib.pyplot as plt
import numpy as np


labels = ['Precision',  'Recall', 'F-score']

baseline = [96.1, 69.2, 77.1]
abra = [91.5, 79.6, 84.5]
own = [99.1, 75.6, 83.7]


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
ax.set_title('Models compared for Validation set 1850 to 1845')
ax.set_xticks([r + width for r in range(len(baseline))], labels)
ax.set_ylim([40, 105])
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)
ax.bar_label(rects3, padding=3)

plt.grid(True)
fig.tight_layout()
plt.savefig(
    'Experiments/plots/data/bar_5_EM_All_1850_1845.png')
plt.show()
