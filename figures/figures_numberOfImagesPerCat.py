import os
import glob

import numpy as np

import matplotlib.pyplot as plt

plt.rcParams["savefig.bbox"] = 'tight'


if __name__ == '__main__':
    # 
    saving_directory = 'figures'
    fig_name = 'numberOfImagesPerCat'
    fig_extensions = ['.png', '.svg']

    root_directory = 'dataset_chat'


    # Lecture des directory
    subdirectories = [d for d in os.listdir(root_directory) if os.path.isdir(os.path.join(root_directory, d))]

    images_per_cat = []

    for subdirectory in subdirectories:
        subdirectory_path = os.path.join(root_directory, subdirectory)
            
        jpg_files = glob.glob(os.path.join(subdirectory_path, "*.jpg"))

        images_per_cat.append(len(jpg_files))

    images_per_cat.sort(reverse=True)


    # Figure
    fig, ax = plt.subplots(1, 1, figsize=(4, 2.5))

    ax.set_xlabel('n-ième chat')
    ax.set_ylabel('Images par chat')
    ax.set_xlim(0, len(images_per_cat))
    ax.set_ylim(0, 100)

    ax.plot(images_per_cat, '.')

    plt.grid(True)
    fig.tight_layout()

    for extension in fig_extensions:
        filename = os.path.join(saving_directory, fig_name + extension)
        fig.savefig(filename, dpi = 600)

    plt.show()
