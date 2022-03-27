import matplotlib.pyplot as plt
import numpy as np


labels = ['Junget', 'Thy early',
          'Thy late', 'Val. early',  'Val. late']
baseline_3 = [100, 95.5, 95, 87.7, 90.4]


plt.bar(labels, baseline_3)
plt.title('Baseline: Set size after conflicts are removed')
plt.xlabel('Test set', fontsize=14)
plt.ylabel('Percentage', fontsize=14)
plt.ylim([0, 108])
plt.grid(True)
plt.savefig('Experiments/plots/data/bar_plot_conflict_Baseline.png')
# plt.show()

# x = np.arange(len(labels))  # the label locations
# width = 0.35  # the width of the bars


# fig, ax = plt.subplots()
# rects1 = ax.bar(x + width, baseline_3, width,
#                 label='3 features', align="center")

# ax.set_ylabel('Percentage')
# ax.set_title('Baseline: Set size after conflicts are removed')
# ax.set_xticks(x, labels)
# ax.set_ylim([0, 108])
# ax.legend(loc='lower right')

# ax.bar_label(rects1, padding=3)

# fig.tight_layout()

# # plt.savefig('Experiments/plots/data/bar_plot_conflict_Baseline.png')
# plt.show()
