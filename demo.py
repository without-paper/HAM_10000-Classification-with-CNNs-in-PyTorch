import os
import random
import torch
from torchvision import transforms
from PIL import Image
from models.ResNet_model101 import ResNet101  # Import your model definition
import matplotlib.pyplot as plt  # Import for displaying images
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", message=".*pretrained.*")
warnings.filterwarnings("ignore", message=".*Arguments other than a weight enum.*")

# Load model architecture
device = torch.device("cpu")
model = ResNet101(dropout_prob=0.5)

# Load state dictionary
model.load_state_dict(torch.load('pth/resnet101_model.pth', map_location=device))
model = model.to(device)
model.eval()  # Set to evaluation mode

# Image preprocessing (based on training process)
data_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.7633, 0.5458, 0.5704], std=[0.09, 0.1188, 0.1334])
])

# Define class names and map folder names to full names
class_names = {
    'akiec': 'ACTINIC KERATOSIS',
    'bcc': 'BASAL CELL CARCINOMA',
    'bkl': 'BENIGN KERATOSIS-LIKE LESIONS',
    'df': 'DERMATOFIBROMA',
    'mel': 'MELANOMA',
    'nv': 'MELANOCYTIC NEVI',
    'vasc': 'VASCULAR LESIONS'
}

# Randomly select four images from the test folder
def get_random_images():
    test_path = 'eval'
    images = []
    for _ in range(4):
        category = random.choice(os.listdir(test_path))  # Randomly choose a subfolder
        img_folder = os.path.join(test_path, category)
        img_name = random.choice(os.listdir(img_folder))  # Randomly select an image
        img_path = os.path.join(img_folder, img_name)
        images.append((img_path, category))
    return images

# Predict the class of an image
def predict_image(img_path):
    image = Image.open(img_path).convert('RGB')
    image = data_transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)  # Load image onto the CPU
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return list(class_names.values())[predicted.item()]

# Run demo
def run_demo():
    while True:
        images = get_random_images()

        # Display the selected four images with real labels
        print("Please select an image:")
        for idx, (img_path, category) in enumerate(images):
            img = Image.open(img_path)
            plt.imshow(img)
            plt.axis('off')
            plt.title(f"Image {idx + 1} - {class_names[category]}")
            plt.show()
            # Use class_names dictionary to map folder name to full name
            print(f"{idx + 1}. Real Type: {class_names[category]}")


        choice = -1  # Initialize with an invalid choice
        while choice not in [1, 2, 3, 4]:
            try:
                choice = int(input("Choose the image number (1-4) for model classification: "))
                if choice not in [1, 2, 3, 4]:
                    print("Invalid choice. Please enter a number between 1 and 4.")
            except ValueError:
                print("Invalid input. Please enter a valid integer between 1 and 4.")

        choice -= 1  # Adjust to 0-based index


        selected_img_path, real_category = images[choice]

        # Print real category
        print(f"Real Category is: {class_names[real_category]}")

        # Model prediction
        predicted_category = predict_image(selected_img_path)
        print(f"Model Prediction: {predicted_category}")

        # Assess correctness
        if class_names[real_category] == predicted_category:
            print("ResNet101: \"Great! I'm glad I'm correct :)\"")
        else:
            print("ResNet101: \"Oops :( I hope I can do better next time\"")

        # Continue or exit
        continue_choice = input("Would you like to continue? (y/n): ")
        if continue_choice.lower() != 'y':
            print("Thank you for trying the demo!")
            break

if __name__ == "__main__":
    run_demo()
