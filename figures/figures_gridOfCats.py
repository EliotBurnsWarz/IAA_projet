import os
import glob

import numpy as np

from PIL import Image

import matplotlib.pyplot as plt

plt.rcParams["savefig.bbox"] = 'tight'


if __name__ == '__main__':
    # Params
    saving_directory = 'figures'
    fig_name = 'gridOfCats'
    # fig_extensions = ['png']  # test number_of_cats
    fig_extensions = ['png', 'svg']  # pour saver
    number_of_cats = 10

    # À décommenter au besoin
    root_directory = 'dataset_chat'
    # subdirectory = '0001'
    # subdirectory = '0099'
    subdirectory = '0171'

    # root_directory = 'ourOwnDataset'
    # subdirectory = 'A0001'


    # It
    subdirectory_path = os.path.join(root_directory, subdirectory)
    jpg_files = glob.glob(os.path.join(subdirectory_path, "*.jpg"))

    list_of_imgs = [Image.open(jpg_file) for jpg_file in jpg_files[0:number_of_cats]]

    horizontal_gid = np.hstack(list_of_imgs)


    # Figure
    fig, ax = plt.subplots(1, 1, figsize=(6, 2))

    # for ax, img in zip(axs, list_of_imgs):
    ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    #     ax.set_aspect('equal')

    ax.imshow(horizontal_gid)

    fig.tight_layout()

    for extension in fig_extensions:
        filename = os.path.join(saving_directory, 
            f"{fig_name}_{subdirectory}.{extension}") 
        
        fig.savefig(filename, dpi = 300)

    plt.show()
