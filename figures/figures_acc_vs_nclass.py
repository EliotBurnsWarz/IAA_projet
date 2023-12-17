import os
import glob
import numpy as np

import matplotlib.pyplot as plt





saving_directory = 'figures'
fig_name = 'acc_vs_nclass'
# fig_extensions = ['png']  # test number_of_cats
fig_extensions = ['png', 'svg']  # pour saver


# nclasses = [2, 4, 8, 16, 32, 64, 128, 256, 434]
nclasses = [8, 16, 32, 64, 128, 256, 434]


fig, ax = plt.subplots(1, 1, figsize=(5, 3))

ax.set_xlabel('Époch')
ax.set_ylabel('Précision (%)')

color = [f"C{i}" for i in range(len(nclasses))]

for i, nclass in enumerate(nclasses):

    filename = f"mobilenet_v2_{nclass}outputs_8epochs.csv"

    with open(filename, 'r') as f:
        data = np.genfromtxt(f, delimiter=",")
    
    acc_train = data[0, :]
    acc_val = data[1, :]

    ax.plot(acc_train, '-', color = color[i], label = str(nclass))
    # ax.plot(acc_val, '--', color = color[i])
    
ax.legend(loc = 'best')

fig.tight_layout()

for extension in fig_extensions:
    filename = os.path.join(saving_directory, 
        f"{fig_name}.{extension}") 
    
    fig.savefig(filename, dpi = 300)

plt.show()