import os
import random
import torch
from torchvision import transforms
from PIL import Image
from flask import Flask, render_template, request, send_from_directory
from models.ResNet_model101 import ResNet101
import warnings
from werkzeug.utils import secure_filename

# Suppress specific warnings
warnings.filterwarnings("ignore", message=".*pretrained.*")
warnings.filterwarnings("ignore", message=".*Arguments other than a weight enum.*")

# Flask app setup
app = Flask(__name__, template_folder="templates", static_folder="eval")  # Set static_folder to 'eval'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Function to check allowed image extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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

        real_category_name = class_names.get(category, "Unknown Category")
        images.append((img_path, real_category_name))
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

# Home route to display images and buttons
@app.route("/", methods=["GET", "POST"])
def home():
    images = get_random_images()
    selected_img_path = None
    real_category_name = None
    predicted_category = None
    feedback = None

    if request.method == "POST":
        # Check if an image is chosen or a file is uploaded
        if 'image_choice' in request.form:
            choice = int(request.form["image_choice"])  # Get selected image index
            selected_img_path, real_category_name = images[choice]  # Get real category name
            predicted_category = predict_image(selected_img_path)
            feedback = "ResNet101: Great! I'm glad I'm correct :)" if real_category_name == predicted_category else "ResNet101: Oops :( I hope I can do better next time"

        elif 'image_file' in request.files:
            # Handle when user uploads their own image
            file = request.files['image_file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join('uploads', filename)
                file.save(file_path)

                # Get the real category from the user input
                real_category_name = request.form.get('real_category')

                if not real_category_name:
                    feedback = "Please enter the real category name for the uploaded image."
                else:
                    # Predict the uploaded image's category
                    predicted_category = predict_image(file_path)
                    feedback = "ResNet101: Great! I'm glad I'm correct :)" if real_category_name == predicted_category else "ResNet101: Oops :( I hope I can do better next time"

    # Convert image paths to URLs for web display
    image_urls = [f"/eval/{os.path.basename(os.path.dirname(img[0]))}/{os.path.basename(img[0])}" for img in images]

    return render_template("index.html", images=images, image_urls=image_urls, feedback=feedback,
                           real_category=real_category_name, predicted_category=predicted_category)


# Serve image files directly from eval folder
@app.route("/eval/<folder>/<filename>")
def serve_image(folder, filename):
    return send_from_directory(os.path.join('eval', folder), filename)

if __name__ == "__main__":
    app.run(debug=True)
