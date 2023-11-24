from PIL import Image
import torchvision.transforms as transforms


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

