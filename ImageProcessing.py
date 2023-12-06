from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


def preprocess_image(image_path):
    image = Image.open(image_path)  
    IMG_SIZE = 224
    preprocess = transforms.Compose([
                    transforms.Resize((IMG_SIZE, IMG_SIZE)),
                    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize pixel values (mean and std values are common for many pre-trained models)
                    ])
    image = preprocess(image)
    return image

def display_train_and_test_images(train_images, test_images, required_train_imgs, transpose):
    fig, axs = plt.subplots(10, 5, figsize=(10, 20))

    for i in range(10):
        # Iterate over columns
        for j in range(4):

            image_index = i * (required_train_imgs) + j
            image_tensor = train_images[image_index]
            image_array = np.transpose(image_tensor, transpose)

            # Display the image
            axs[i, j].imshow(image_array) 
            axs[i, j].axis('off')  # Optional: Turn off axis labels
            
            if i == 0:
                axs[i, j].set_title('Train Images')
            
        # Display the image
        axs[i, 4].imshow(np.transpose(test_images[i], transpose))
        axs[i, 4].axis('off')  # Optional: Turn off axis labels
        
        if i == 0:
            axs[i, 4].set_title('Test Image')
        
    plt.tight_layout()
    plt.show()

