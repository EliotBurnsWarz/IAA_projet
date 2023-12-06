import os
import glob

import numpy as np

from PIL import Image, ImageFilter

import torch

import torchvision.transforms.functional as F

import torchvision.transforms as transforms
from torchvision.models.segmentation import fcn_resnet50

from torchvision.ops import masks_to_boxes

from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

import matplotlib.pyplot as plt

plt.rcParams["savefig.bbox"] = 'tight'

import time





def preprocess_original_image(original_filepath, directory_for_background, directory_for_no_background,
                              print_when_done = True, final_dimension = 244, resize_dim_for_fasterrcnn = 500,
                              save_with_background = True, show_final_image_with_background = False,
                              save_without_background = True, show_final_image_without_background = False,
                              save_with_background_as_tensor = False, save_without_background_as_tensor = False, 
                              show_mask_cat = False, show_mask_background = False, show_box = False,
                              show_image_without_background = False):
    
    start_time = time.time()

    # original_directory, subdirectory, filename = original_filepath.split('/')
    subdirectory, filename = os.path.split(original_filepath)
    
    # Get the directory name from the directory path
    subdirectory = os.path.basename(subdirectory)

    # file_directory_with_background = directory_for_background + '/' + subdirectory
    # file_directory_without_background = directory_for_no_background + '/' + subdirectory
    file_directory_with_background = os.path.join(directory_for_background, subdirectory)
    file_directory_without_background = os.path.join(directory_for_no_background, subdirectory)

    # filepath_with_background = file_directory_with_background + '/' + filename
    # filepath_without_background = file_directory_without_background + '/' + filename
    filepath_with_background = os.path.join(file_directory_with_background, filename)
    filepath_without_background = os.path.join(file_directory_without_background, filename)

    # Image
    with Image.open(original_filepath) as im:
        square_image = pad_image_to_make_square(im)

    square_image_dim = square_image.width

    scaling_factor = square_image_dim/resize_dim_for_fasterrcnn
    resized_image = square_image.resize((resize_dim_for_fasterrcnn, resize_dim_for_fasterrcnn))

    # Tenseur
    resized_tensor_0_255, normalized_tensor = image_to_tensors(resized_image)

    try:
        square_box, mask_cat, _ = box_and_masks_with_fcn_resnet50(normalized_tensor)
    except Exception as e:
        #move file
        os.rename(original_filepath, os.path.join('dataset_preprocessing_errors', filename))
        return None  # on quitte la fonction
    
    mask_background = ~mask_cat

    left, upper, right, lower = get_int_corner_from_box_tensor(scaling_factor*square_box)

    if save_with_background:
        im_crop = square_image.crop((left, upper, right, lower))
        im_crop_resized = im_crop.resize((final_dimension, final_dimension))

        # On crée le/les dossiers s'ils n'existent pas
        try:
            # os.makedirs(os.path.abspath(os.getcwd()) + '\\' + file_directory_with_background.replace("/", "\\"))  # créer le dossier
            os.makedirs(os.path.join(os.path.abspath(os.getcwd()), file_directory_with_background))  # créer le dossier
        except (OSError, FileExistsError) as error: 
            pass

        # SAVE
        im_crop_resized.save(filepath_with_background)

        if save_with_background_as_tensor:
            _, normalized_tensor = image_to_tensors(im_crop_resized)

            filepath, _ = filepath_with_background.rsplit('.', 1)

            torch.save(normalized_tensor, filepath + '.pt')

        if show_final_image_with_background:
            im_crop_resized.show()

    if save_without_background:
        mask_as_img = F.to_pil_image(mask_cat.float())
        mask_as_img = mask_as_img.filter(ImageFilter.GaussianBlur(radius=2))
        upscaled_mask_as_img = mask_as_img.resize((square_image_dim, square_image_dim))
    
        black_image = Image.new('RGB', size = (square_image_dim, square_image_dim), color = 'black')
        image_no_background = Image.composite(square_image, black_image, upscaled_mask_as_img)

        image_no_background_cropped = image_no_background.crop((left, upper, right, lower))
        im_no_background_crop_resized = image_no_background_cropped.resize((final_dimension, final_dimension))

        # On crée le/les dossiers s'ils n'existent pas
        try:
            # os.makedirs(os.path.abspath(os.getcwd()) + '\\' + file_directory_without_background.replace("/", "\\"))  # créer le dossier
            os.makedirs(os.path.join(os.path.abspath(os.getcwd()), file_directory_without_background))  # créer le dossier
        except (OSError, FileExistsError) as error: 
            pass

        # SAVE
        im_no_background_crop_resized.save(filepath_without_background)

        if save_without_background_as_tensor:
            _, normalized_tensor = image_to_tensors(im_no_background_crop_resized)

            filepath, _ = filepath_without_background.rsplit('.', 1)

            torch.save(normalized_tensor, filepath + '.pt')

        if show_final_image_without_background:
            im_no_background_crop_resized.show()

    # AFFICHAGE
    if show_mask_cat:
        show(mask_cat.float()) 
    
    if show_mask_background:
        show(mask_background.float())

    if show_box:
        drawn_boxes = draw_bounding_boxes(resized_tensor_0_255, square_box, colors="red")
        show(drawn_boxes)

    if show_image_without_background:
        dogs_with_masks = draw_segmentation_masks(resized_tensor_0_255, masks = mask_background, alpha=1, colors = 'black')#'white')
        show(dogs_with_masks)

    if print_when_done:
        print(original_filepath + ' -> Done in ' + str(np.round(time.time() - start_time, 2)) + ' s!')

    # ROTATION -> pour data augmentation

    # im.rotate(45).show()

    # im.rotate(angle = 45, expand = True, fillcolor = 'white').show()


def pad_image_to_make_square(rectangle_image, fill_color = 'black'):
    width, height = rectangle_image.size

    max_dim = np.max([width, height])

    square_image = Image.new('RGB', size = (max_dim, max_dim), color = fill_color)
    square_image.paste(rectangle_image, (int((max_dim - width) / 2), int((max_dim - height) / 2)))

    return square_image


def denormalize(normalized_tensor_0_1):
    """
    Pour passer d'une représentation 0-1 en float à une
    représentation 0-255 en int8
    """
    denormalize_tensor_0_255 = (normalized_tensor_0_1*255).to(torch.uint8)

    return denormalize_tensor_0_255


def image_to_tensors(image):
    to_tensor = transforms.Compose([transforms.ToTensor()])
    tensor_0_1 = to_tensor(image)
    tensor_0_255 = (tensor_0_1*255).to(torch.uint8)
    normalize_as_ImageNet = transforms.Compose([
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])

    normalized_tensor = normalize_as_ImageNet(tensor_0_1)
    return tensor_0_255, normalized_tensor


def box_and_masks_with_fcn_resnet50(normalized_tensor):
    # On utilise un réseau pré entrainé pour trouver le chat
    model = fcn_resnet50(weights = 'DEFAULT', progress = False)
    model.eval()

    output = model(normalized_tensor.unsqueeze(0))['out']

    normalized_masks = torch.nn.functional.softmax(output, dim=1)

    # les différentes classes en sortie
    sem_classes = [
        '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
        'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]

    sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}

    class_dim = 1

    ## il détecte des parties comme étant des chiens, si on les inclut
    # pas, on se retrouve à avoir des masques qu'il manque des bouts
    # de pattes ou d'oreille

    # par contre, si on prend le chien en plus, on peut avoir des éléments
    # par rapport qui s'ajoute... donc pour le masque, c'est utile, mais
    # pour la boite, il vaut mieux faire la boite autour du masque 'chat'
    # uniquement...
    
    boolean_cat_mask = normalized_masks.argmax(class_dim) == sem_class_to_idx['cat']

    boolean_cat_dog_mask = torch.logical_or(normalized_masks.argmax(class_dim) == sem_class_to_idx['cat'],
                                    normalized_masks.argmax(class_dim) == sem_class_to_idx['dog'])

    mask_cat = boolean_cat_mask 
    mask_cat_dog = boolean_cat_dog_mask 

    rectangle_box = masks_to_boxes(boolean_cat_mask)
    square_box = rectangle_box_to_square_box(rectangle_box)

    return square_box, mask_cat_dog, mask_cat


def rectangle_box_to_square_box(rectangle_box):
    x1, y1, x2, y2 = rectangle_box.squeeze()

    rectangle_box_width = x2 - x1
    rectangle_box_height = y2 - y1

    box_center_x = (x1 + x2)/2
    box_center_y = (y1 + y2)/2

    rectangle_box_max_dim = max(rectangle_box_width, rectangle_box_height)

    # Il n'y a rien qui assure que x1 et y1 sont > 0,
    # ni que x2 et y2 sont < im.width
    square_box_x1 = box_center_x - rectangle_box_max_dim/2
    square_box_x2 = box_center_x + rectangle_box_max_dim/2
    
    square_box_y1 = box_center_y - rectangle_box_max_dim/2
    square_box_y2 = box_center_y + rectangle_box_max_dim/2

    square_box = torch.tensor([square_box_x1, square_box_y1, square_box_x2, square_box_y2]).unsqueeze(0)

    return square_box


def get_int_corner_from_box_tensor(box_tensor):
    left, upper, right, lower = torch.round(box_tensor).squeeze()

    left = int(left.item())
    upper = int(upper.item())
    right = int(right.item())
    lower = int(lower.item())

    return left, upper, right, lower


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]

    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)

    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def show_images_as_grid(list_of_imgs, image_per_line = 3):
    if not isinstance(list_of_imgs, list):
        list_of_imgs = [list_of_imgs]

    N = len(list_of_imgs)

    ncols = image_per_line
    nrows = 1 + (N - 1) // image_per_line

    if nrows == 1:
        ncols = len(list_of_imgs)

    fig, axs = plt.subplots(nrows = nrows, ncols = ncols, figsize = (ncols*2.1, nrows*2.1))
    plt.subplots_adjust(wspace=0, hspace=0)

    for axs_row in axs:
        for ax in axs_row:
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            ax.set_aspect('equal')

    for i, img in enumerate(list_of_imgs):
        row, col = divmod(i, ncols)
        ax = axs[row, col]

        img = img.detach()
        img = F.to_pil_image(img)
        
        ax.imshow(np.asarray(img))










if __name__ == '__main__':

    root_directory = "selected_datasets/selected_dataset_chat"

    subdirectories = [d for d in os.listdir(root_directory) if os.path.isdir(os.path.join(root_directory, d))]

    for subdirectory in subdirectories:
        subdirectory_path = os.path.join(root_directory, subdirectory)
            
        jpg_files = glob.glob(os.path.join(subdirectory_path, "*.jpg"))

        print(str(len(jpg_files)) + ' files in the subdirectory ' + subdirectory)
        for jpg_file in jpg_files:
            preprocess_original_image(jpg_file, 
                directory_for_background = "selected_datasets/prep_bounding_box/", 
                directory_for_no_background =  "selected_datasets/prep_no_back/")

        print('Done!')


