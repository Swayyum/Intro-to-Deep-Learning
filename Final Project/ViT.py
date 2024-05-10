# %%
import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import torch
import torch.nn as nn
import timm
from PIL import Image
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import seaborn as sns
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class VisionTransformerForSegmentation(nn.Module):
    def __init__(self, img_size, num_classes):
        super(VisionTransformerForSegmentation, self).__init__()
        # Load a pre-trained Vision Transformer model
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        self.decoder = nn.Sequential(
            nn.Conv2d(768, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, num_classes, kernel_size=1)
        )
        self.img_size = img_size

    def forward(self, x):
        # Reshape x to fit ViT input
        B, C, H, W = x.shape
        x = self.vit(x)  # (B, num_patches, embedding_dim)
        x = x.permute(0, 2, 1).view(B, -1, int(H / 16), int(W / 16))  # reshape to feature map
        x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        x = self.decoder(x)
        return x


class PKLotDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith('_img.npy')]
        self.transform = transform
        print(f"Found {len(self.image_files)} images in directory {data_dir}")

    def __len__(self):
        return len(self.image_files)  # This should return the number of items in the dataset

    def __getitem__(self, idx):
        image_file = os.path.join(self.data_dir, self.image_files[idx])
        mask_file = image_file.replace('_img.npy', '_mask.npy')

        image = np.load(image_file)
        mask = np.load(mask_file)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


# Define the evaluation function
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_pixels = 0

    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            preds = outputs > 0.5
            total_loss += loss.item() * images.size(0)
            total_correct += preds.eq(masks).sum().item()
            total_pixels += masks.numel()

    average_loss = total_loss / len(loader.dataset)
    accuracy = total_correct / total_pixels
    return average_loss, accuracy


dataset_base = r'C:/Users/SirM/Desktop/Swayam/Intro to Deep Learning/Intro-to-Deep-Learning/Final Project/PKLot.v1-raw.yolov8-obb'
partitions = ['train', 'valid', 'test']
target_size = (512, 512)

# Create datasets and dataloaders for train, validation, and test sets
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = PKLotDataset(os.path.join(dataset_base, 'train', 'images'), transform=transform)
valid_dataset = PKLotDataset(os.path.join(dataset_base, 'valid', 'images'), transform=transform)
test_dataset = PKLotDataset(os.path.join(dataset_base, 'test', 'images'), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


# # Setup device, model, loss function, and optimizer
# model = CarNet().to(device)
# criterion = DiceLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

def calculate_accuracy(outputs, masks, threshold=0.5):
    # Apply sigmoid to get probabilities
    probs = torch.sigmoid(outputs)
    # Apply threshold to get binary tensor
    preds = (probs > threshold).float()
    # Calculate accuracy
    correct = (preds == masks).float()
    accuracy = correct.sum() / correct.numel()
    return accuracy.item()


# Initialize lists to store accuracies for each batch
accuracies = []
train_losses = []
valid_losses = []
valid_accuracies = []
train_accuracies = []
start_time = time.time()
# Training loop
for epoch in range(50):
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        train_correct += (preds == masks).float().sum().item()
        train_total += masks.numel()

    train_losses.append(train_loss / len(train_loader.dataset))
    train_accuracies.append(train_correct / train_total)

    # Validation step
    model.eval()
    valid_loss = 0
    valid_correct = 0
    valid_total = 0
    with torch.no_grad():
        for images, masks in valid_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)

            valid_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            valid_correct += (preds == masks).float().sum().item()
            valid_total += masks.numel()

    valid_losses.append(valid_loss / len(valid_loader.dataset))
    valid_accuracies.append(valid_correct / valid_total)

    print(
        f'Epoch {epoch + 1}: Train Loss = {train_losses[-1]:.4f}, Val Loss = {valid_losses[-1]:.4f}, Train Acc = {train_accuracies[-1]:.4f}, Val Acc = {valid_accuracies[-1]:.4f}')

# Plotting training vs validation loss
plt.figure(figsize=(12, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(valid_losses, label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plotting training vs validation accuracy
plt.figure(figsize=(12, 6))
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(valid_accuracies, label='Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Save the trained model
torch.save(model.state_dict(), 'car_segmentation_model1.pth')

# Load the trained model for testing
model.load_state_dict(torch.load('car_segmentation_model1.pth'))
model.to(device)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_pixels = 0
    all_preds = []
    all_true = []

    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            preds = (torch.sigmoid(outputs) > 0.5).float()

            total_loss += loss.item() * images.size(0)
            total_correct += (preds == masks).float().sum().item()
            total_pixels += masks.numel()

            all_preds.append(preds.view(-1).cpu().numpy())
            all_true.append(masks.view(-1).cpu().numpy())

    average_loss = total_loss / len(loader.dataset)
    accuracy = total_correct / total_pixels
    all_preds = np.concatenate(all_preds)
    all_true = np.concatenate(all_true)
    precision, recall, f1, _ = precision_recall_fscore_support(all_true, all_preds, average='binary')
    cm = confusion_matrix(all_true, all_preds)

    return average_loss, accuracy, precision, recall, f1, cm


# Calculate test metrics and plot confusion matrix
test_loss, test_accuracy, test_precision, test_recall, test_f1, test_cm = evaluate(model, test_loader, criterion,
                                                                                   device)
print(
    f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1 Score: {test_f1:.4f}")


def plot_confusion_matrix(cm, classes, title='Confusion Matrix'):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


plot_confusion_matrix(test_cm, classes=['Background', 'Object'], title='Test Confusion Matrix')

end_time = time.time()
inference_time = end_time - start_time
print(f"Total Time taken for inference: {inference_time:.4f} seconds")

# %%
model_path = 'car_segmentation_model.pth'
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()  # Set the model to evaluation mode fo


def visualize_predictions(model, loader, device, threshold=0.5, num_images_per_batch=6, num_batches=6):
    model.eval()  # Set the model to evaluation mode
    images_processed = 0
    batches_processed = 0

    with torch.no_grad():  # Disable gradient computation
        for images, true_masks in loader:
            if batches_processed >= num_batches:
                break  # Stop after the desired number of batches

            images, true_masks = images.to(device), true_masks.to(device)
            preds = model(images)
            preds = torch.sigmoid(preds) > threshold  # Apply threshold to get binary predictions

            # Visualize the specified number of images from the current batch
            for i in range(min(num_images_per_batch, images.size(0))):
                fig, axs = plt.subplots(1, 3, figsize=(20, 6))

                # Original Image
                img = images[i].cpu().permute(1, 2, 0).numpy()
                img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0,1]
                axs[0].imshow(img)
                axs[0].set_title('Original Image')
                axs[0].axis('off')

                # True Mask
                axs[1].imshow(true_masks[i].cpu().squeeze(), cmap='gray')
                axs[1].set_title('True Mask')
                axs[1].axis('off')

                # Predicted Mask
                axs[2].imshow(preds[i].cpu().squeeze(), cmap='gray')
                axs[2].set_title('Predicted Mask')
                axs[2].axis('off')

                plt.show()
                images_processed += 1

            batches_processed += 1

    print(f"Displayed {images_processed} images from {batches_processed} batches.")


# Example usage
visualize_predictions(model, test_loader, device, threshold=0.5, num_images_per_batch=6, num_batches=6)
