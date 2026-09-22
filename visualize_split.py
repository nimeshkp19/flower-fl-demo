"""
Usage:
    python visualize_split.py            # 10 clients (default, matches the demo)
    python visualize_split.py --clients 3
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from fl_demo.task import get_partition_label_counts

parser = argparse.ArgumentParser()
parser.add_argument("--clients", type=int, default=10)
args = parser.parse_args()

NUM_CLIENTS = args.clients
NUM_CLASSES = 10

counts = get_partition_label_counts(NUM_CLIENTS)

# Build a (clients x classes) matrix of counts
matrix = np.zeros((NUM_CLIENTS, NUM_CLASSES))
for pid, counter in counts.items():
    for digit, n in counter.items():
        matrix[pid, digit] = n

fig, ax = plt.subplots(figsize=(9, 5))
bottom = np.zeros(NUM_CLIENTS)
colors = plt.cm.tab10(np.linspace(0, 1, NUM_CLASSES))

for digit in range(NUM_CLASSES):
    ax.bar(
        [f"Client {i}" for i in range(NUM_CLIENTS)],
        matrix[:, digit],
        bottom=bottom,
        label=str(digit),
        color=colors[digit],
    )
    bottom += matrix[:, digit]

ax.set_ylabel("Number of training images")
ax.set_title(
    f"Each client only sees {2} digit classes\n"
    "(non-IID split - no single client could learn all 10 digits alone)"
)
ax.legend(title="Digit", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig("non_iid_split.png", dpi=150)
print("Saved chart to non_iid_split.png")
plt.show()
