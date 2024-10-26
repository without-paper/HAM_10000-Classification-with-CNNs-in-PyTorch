import pandas as pd 
import shutil
import os

# Read the CSV file
df = pd.read_csv("HAM10000_images\HAM10000_metadata.csv")

# Iterate through each row
for index, row in df.iterrows():
    image_id = row['image_id']  # Replace with the actual image ID column name
    label = row['dx']  # Replace with the actual label column name

    # Construct the full path to the image file
    image_path = os.path.join("HAM10000_images", image_id + ".jpg")  # Assuming images have a .jpg extension

    # Create the target folder
    target_folder = os.path.join("HAM10000_images", str(label))
    os.makedirs(target_folder, exist_ok=True)

    # Copy or move the image to the target folder
    shutil.copy(image_path, target_folder)  # If you want to move instead of copying, use shutil.move
