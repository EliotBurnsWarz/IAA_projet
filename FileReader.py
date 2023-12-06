import os 
import torch
import glob

from ImageProcessing import preprocess_image

def get_picture_tensors(root_directory, 
                        n_classes, 
                        required_train_imgs, 
                        required_test_imgs, 
                        use_validation = True,
                        use_selected_eval_datasets = False):
    
    subdirectories = [d for d in os.listdir(root_directory) if os.path.isdir(os.path.join(root_directory, d))]

    # créer des listes vides
    train_images, val_images, test_images, train_labels, val_labels, test_labels = [[] for _ in range(6)]

    if (use_validation):
        # Ajouter données pour la validation (autant que test)
        required_images_per_cat = required_train_imgs + 2 * required_test_imgs
    else: required_images_per_cat = required_train_imgs + required_test_imgs

    current_index = 0
    max_index = len(subdirectories)
    current_amount_of_classes = 0
    while current_amount_of_classes < n_classes and current_index < max_index:
        subdirectory = subdirectories[current_index]
        subdirectory_path = os.path.join(root_directory, subdirectory)
        jpeg_files = glob.glob(os.path.join(subdirectory_path, "*.jpg"))
            
        if len(jpeg_files) < required_images_per_cat:
            print(f"{subdirectory_path} does not contain enough images, will not be used")
        else:
            
            # créer des one-hots pour labels
            label = torch.zeros(n_classes)
            label[current_amount_of_classes] = 1
                
            print('Chargement de ' + subdirectory_path + '  ->  ' + str(required_images_per_cat) + '/' + str(len(jpeg_files)) + ' images')
            
            if use_selected_eval_datasets:
                if os.path.exists(os.path.join(subdirectory_path, "T.jpg")) and os.path.exists(os.path.join(subdirectory_path, "V.jpg")) :
                    val_path = jpeg_files[0][:35] + "\\T.JPG"
                    test_path = jpeg_files[0][:35] + "\\V.JPG"
                    t_index = jpeg_files.index(val_path)
                    v_index = jpeg_files.index(test_path)
                    
                    exclude_indices = [t_index, v_index]
                    filtered_files = [file for i, file in enumerate(jpeg_files) if i not in [t_index, v_index]]

                    add_image_to_lists(filtered_files[:required_train_imgs], train_images, train_labels, label)

                    add_image_to_lists([test_path], test_images, test_labels, label)

                    if (use_validation):
                        add_image_to_lists([val_path], val_images, val_labels, label)
                        
                    


                else:
                    print(f"The file {subdirectory_path} does not have validation or test .")
                    
                
            
            else:
               
                # train
                add_image_to_lists(jpeg_files[:required_train_imgs], train_images, train_labels, label)

                # test
                add_image_to_lists(jpeg_files[required_train_imgs:required_train_imgs+required_test_imgs], test_images, test_labels, label)

                if (use_validation):
                    add_image_to_lists(jpeg_files[required_train_imgs+required_test_imgs:required_train_imgs+2*required_test_imgs], val_images, val_labels, label)
                

            current_amount_of_classes += 1
            
        current_index += 1
            
            

    if current_amount_of_classes < n_classes:
        print(f"Could not find {n_classes} valid classes in dataset, only {current_amount_of_classes} classes will be used")
        n_classes_to_use = current_amount_of_classes
    else:
        n_classes_to_use = n_classes

    print('Done!')
    

    return train_images, val_images, test_images, train_labels, val_labels, test_labels, n_classes_to_use



    
def add_image_to_lists(jpeg_file_list, image_list, label_list, label):
    for jpeg_file in jpeg_file_list:
        tensor_image = preprocess_image(jpeg_file)
        image_list.append(tensor_image)
        label_list.append(label)
        