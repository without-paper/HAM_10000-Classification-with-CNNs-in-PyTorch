import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import os
from PIL import Image
import multiprocessing
from logger import setup_logger
import random

from models.ResNet_model18 import ResNet18
from models.ResNet_model34 import ResNet34
from models.ResNet_model50 import ResNet50
from models.ResNet_model101 import ResNet101
from models.ResNet_model152 import ResNet152
from models.vit_model import ViT
from models.AlexNet_model import AlexNet
from models.mlp_model import MlpMixer

# Set up logger to save logs to 'training.log' file
logger = setup_logger()


if __name__ == '__main__':
    multiprocessing.freeze_support()

    # Define the root directory of the image folder
    data_root = "HAM10000_images/train"

    # Get the names of the subfolders for each category
    class_folders = os.listdir(data_root)

    # Initialize lists to store image file paths and labels
    image_paths = []
    labels = []


    # Traverse each category folder
    for label, class_folder in enumerate(class_folders):
        class_path = os.path.join(data_root, class_folder)

        # Get the image file paths under each category folder
        class_images = [os.path.join(class_path, img) for img in os.listdir(class_path)]

        # Add to the overall image file path list
        image_paths.extend(class_images)

        # Add the corresponding labels
        labels.extend([label] * len(class_images))


    class CustomDataset(Dataset):
        def __init__(self, image_paths, labels, transform=None, subset_percentage=0.1):  # You can change the percentage of subset (max of 1)
            self.image_paths = image_paths
            self.labels = labels
            self.transform = transform
            self.subset_percentage = subset_percentage

            # Determine the number of samples based on the subset percentage
            self.num_samples = int(len(self.image_paths) * self.subset_percentage)

            # Select a random subset of samples
            self.selected_indices = random.sample(range(len(self.image_paths)), self.num_samples)

        def __len__(self):
            return len(self.selected_indices)

        def __getitem__(self, idx):
            idx = self.selected_indices[idx]
            image_path = self.image_paths[idx]
            label = self.labels[idx]

            image = Image.open(image_path).convert("RGB")

            if self.transform:
                image = self.transform(image)

            return image, label


    data_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.7633, 0.5458, 0.5704], std=[0.09, 0.1188, 0.1334])
    ])

    custom_dataset = CustomDataset(image_paths, labels, transform=data_transform)

    train_data_path = data_root
    test_data_path = "HAM10000_images/test"

    train_dataset = datasets.ImageFolder(train_data_path, transform=data_transform)
    test_dataset = datasets.ImageFolder(test_data_path, transform=data_transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    models = {
        'alexnet': AlexNet(dropout_prob=0.5),
        'resnet18': ResNet18(dropout_prob=0.5),
        'resnet34': ResNet34(dropout_prob=0.5),
        'resnet50': ResNet50(dropout_prob=0.5),
        'resnet101': ResNet101(dropout_prob=0.5),
        'resnet152': ResNet152(dropout_prob=0.5),
        'vit': ViT(
            image_size=256, patch_size=32,
            num_classes=7, dim=1024,
            depth=6, heads=16, mlp_dim=2048,
            dropout=0.1, emb_dropout=0.1,
        ),
        'mlp_mixer': MlpMixer(in_dim=3, hidden_dim=32,
                              mlp_token_dim=32, mlp_channel_dim=32,
                              patch_size=(7, 7), img_size=(256, 256),
                              num_block=2, num_class=7
                              ),
    }




    model_infos = {model_name: {'train_losses': [], 'test_losses': [], 'accuracies': []} for model_name in models}

    criterion = nn.CrossEntropyLoss()

    num_epochs = 15
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using {device} device")
    logger.info('Start training!')


    optimizers = {model_name: optim.Adam(model.parameters(), lr=0.00005) for model_name, model in models.items()}
    schedulers = {model_name: torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
                  for model_name, optimizer in optimizers.items()}

    # Folder for storing plots
    os.makedirs("plots", exist_ok=True)

    for model_name, model in models.items():
        model = model.to(device)
        scaler = GradScaler()
        optimizer = optimizers[model_name]
        scheduler = schedulers[model_name]
        train_losses, test_losses, accuracies = [], [], []

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels.long())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item()

            scheduler.step()

            model.eval()
            test_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    with autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels.long())
                    test_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            train_loss = running_loss / len(train_loader)
            test_loss /= len(test_loader)
            accuracy = correct / total

            model_infos[model_name]['train_losses'].append(train_loss)
            model_infos[model_name]['test_losses'].append(test_loss)
            model_infos[model_name]['accuracies'].append(accuracy)

            train_losses.append(train_loss)
            test_losses.append(test_loss)
            accuracies.append(accuracy)

            logger.info(f"Model: {model_name}, Epoch {epoch + 1}/{num_epochs}, "
                        f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")


        # Save models
        save_folder = "pth"
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, f"{model_name}_model.pth")
        torch.save(model.state_dict(), save_path)


        # Plot train loss and test loss graphs
        plt.figure(figsize=(15, 5))

        # Plot train loss
        plt.subplot(1, 2, 1)
        for plot_model_name, model_info in model_infos.items():
            plt.plot(range(1, len(model_info['train_losses']) + 1), model_info['train_losses'],
                    label=f'{plot_model_name} Train Loss')

        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Train Loss')
        plt.legend()

        # Plot test loss
        plt.subplot(1, 2, 2)
        for plot_model_name, model_info in model_infos.items():
            plt.plot(range(1, len(model_info['test_losses']) + 1), model_info['test_losses'],
                    label=f'{plot_model_name} Test Loss')

        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Test Loss')
        plt.legend()

        # Adjust layout
        plt.tight_layout()

        # Save the chart
        plt.savefig("plots/all_models_losses.png")
        plt.show()

        # Plot accuracy graph
        plt.figure(figsize=(10, 5))
        for plot_model_name, model_info in model_infos.items():
            plt.plot(range(1, len(model_info['accuracies']) + 1), model_info['accuracies'],
                    label=f'{plot_model_name} Accuracy')

        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig("plots/all_models_accuracies.png")
        plt.show()

        # Add corresponding labels
        labels.extend([label] * len(class_images))
