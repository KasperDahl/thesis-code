import matplotlib.pyplot as plt
import numpy as np


labels = ['Precision',  'Recall', 'F-score']

feat_3 = [94.6, 75.2, 82.2]
feat_5 = [99.1, 75.6, 83.7]


# pre_3 = [94.6, 93.4]
# recall_3 = [75.2, 77]
# fscore_3 = [82.2, 83.3]

# pre_5 = [99.1, 99.4]
# recall_5 = [75.6, 77.9]
# fscore_5 = [83.7, 85.6]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, feat_3, width, label='3 Feature EM')
rects2 = ax.bar(x + width/2, feat_5, width, label='5 Feature EM')


ax.set_ylabel('Percentage', fontsize=18)
#ax.set_title('Conflicts resolved: Val. set  1850 to 1845')
ax.set_xticks(x, labels, fontsize=18)
ax.set_ylim([0, 105])
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)

plt.grid(True)
fig.tight_layout()
plt.savefig(
    'Experiments/plots/data/bar_plot_NO_CONFLICT_1850_1845.png')
plt.show()
