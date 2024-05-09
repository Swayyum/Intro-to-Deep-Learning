# %%
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import seaborn as sns
import time
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class SegmentationModel(nn.Module):
    def __init__(self):
        super(SegmentationModel, self).__init__()
        self.encoder1 = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.encoder2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.encoder3 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))

        self.decoder3 = nn.Sequential(nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), nn.ReLU())
        self.decoder2 = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU())
        self.decoder1 = nn.Sequential(nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid())

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)

        dec3 = self.decoder3(enc3) + enc2  # Skip connection
        dec2 = self.decoder2(dec3) + enc1  # Skip connection
        dec1 = self.decoder1(dec2)
        return dec1


class PKLotDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith('_img.npy')]
        self.transform = transform
        print(f"Found {len(self.image_files)} images in directory {data_dir}")  # Debugging output

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = os.path.join(self.data_dir, self.image_files[idx])
        mask_file = image_file.replace('_img.npy', '_mask.npy')

        image = np.load(image_file)
        mask = np.load(mask_file)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


dataset_base = r'C:/Users/SirM/Desktop/Swayam/Intro to Deep Learning/Intro-to-Deep-Learning/Final Project/PKLot.v1-raw.yolov8-obb'
# dataset_base = r'Data'
partitions = ['train', 'valid', 'test']
target_size = (512, 512)


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred.sigmoid()
        y_pred_flat = y_pred.view(-1)
        y_true_flat = y_true.view(-1)
        intersection = (y_pred_flat * y_true_flat).sum()
        dice_coeff = (2. * intersection + self.smooth) / (y_pred_flat.sum() + y_true_flat.sum() + self.smooth)
        return 1 - dice_coeff


# Create datasets and dataloaders for train, validation, and test sets
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = PKLotDataset(os.path.join(dataset_base, 'train', 'images'), transform=transform)
valid_dataset = PKLotDataset(os.path.join(dataset_base, 'valid', 'images'), transform=transform)
test_dataset = PKLotDataset(os.path.join(dataset_base, 'test', 'images'), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Setup device, model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SegmentationModel().to(device)
print(model)
summary(model, input_size=(3, 512, 512))  # Adjust input size as necessary
criterion = DiceLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

start_time = time.time()
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []


def evaluate(model, loader, criterion, device, full_metrics=False):
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
            preds = (outputs > 0.5).float()  # Convert to binary predictions
            total_loss += loss.item() * images.size(0)
            total_correct += preds.eq(masks).sum().item()
            total_pixels += masks.numel()
            all_preds.extend(preds.view(-1).cpu().numpy())  # Flatten and store predictions
            all_true.extend(masks.view(-1).cpu().numpy())  # Flatten and store true labels

    average_loss = total_loss / len(loader.dataset)
    accuracy = total_correct / total_pixels

    if full_metrics:
        all_preds = np.array(all_preds).astype(int)  # Convert to integer
        all_true = np.array(all_true).astype(int)
        # Debugging output to check the values before computation
        print("Unique values in preds:", np.unique(all_preds))
        print("Unique values in true:", np.unique(all_true))
        precision, recall, f1, _ = precision_recall_fscore_support(all_true, all_preds, average='binary')
        cm = confusion_matrix(all_true, all_preds)
        return average_loss, accuracy, precision, recall, f1, cm

    return average_loss, accuracy


# Training loop
for epoch in range(105):
    model.train()
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

    train_losses.append(loss.item())  # Captures the last training loss of the epoch
    train_accuracies.append((outputs.argmax(1) == masks).float().mean().item())

    val_loss, val_accuracy = evaluate(model, valid_loader, criterion, device)
    val_losses.append(val_loss)  # Append validation loss here
    val_accuracies.append(val_accuracy)

    print(
        f"Epoch {epoch + 1}, Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_accuracies[-1]:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")


def plot_confusion_matrix(cm, classes=['Empty', 'Occupied'], title='Confusion Matrix - Parking Spots'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()


test_loss, test_accuracy, test_precision, test_recall, test_f1, test_cm = evaluate(model, test_loader, criterion,
                                                                                   device, full_metrics=True)
print(
    f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1 Score: {test_f1:.4f}")

# Plot training vs validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot training vs validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plot_confusion_matrix(test_cm)
# Save the trained model
torch.save(model, 'CNN_model.pth')

# Load the trained model for testing
loaded_model = torch.load('CNN_model.pth')
loaded_model.eval()


def check_masks(data_loader):
    for images, masks in data_loader:
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].imshow(images[0].permute(1, 2, 0))  # Adjust permute for your specific case
        axs[0].set_title('Original Image')
        axs[1].imshow(masks[0].squeeze(), cmap='gray')
        axs[1].set_title('Mask')
        plt.show()
        break  # Just show one batch for checking


check_masks(train_loader)  # Check first batch of the training loader


# Visualize the model predictions with confidence scores
def visualize_with_confidence(model, loader, device, num_examples=5, threshold=0.5):
    model.eval()
    indices = torch.randperm(len(loader.dataset))[:num_examples]  # Randomly select indices
    subset = torch.utils.data.Subset(loader.dataset, indices)  # Create a subset based on these indices
    sub_loader = torch.utils.data.DataLoader(subset, batch_size=1)  # Load the subset

    with torch.no_grad():
        for images, masks in sub_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            sigmoid_outputs = torch.sigmoid(outputs)
            preds = (sigmoid_outputs > threshold).float()

            images_np = images.cpu().numpy()
            masks_np = masks.cpu().numpy()
            preds_np = preds.cpu().numpy()
            sigmoid_outputs_np = sigmoid_outputs.cpu().numpy()

            fig, ax = plt.subplots(1, 4, figsize=(20, 5))
            ax[0].imshow(images_np[0].transpose(1, 2, 0))
            ax[0].set_title('Original Image')
            ax[0].axis('off')

            ax[1].imshow(masks_np[0].squeeze(), cmap='gray')
            ax[1].set_title('True Mask')
            ax[1].axis('off')

            ax[2].imshow(preds_np[0].squeeze(), cmap='gray')
            ax[2].set_title('Predicted Mask')
            ax[2].axis('off')

            overlay_image = images_np[0].transpose(1, 2, 0).copy()
            for y in range(0, preds_np[0].shape[1], 32):  # Adjust grid size if needed
                for x in range(0, preds_np[0].shape[2], 32):
                    conf_score = np.mean(sigmoid_outputs_np[0][:, y:y + 32, x:x + 32])
                    status = "Occupied" if conf_score > threshold else "Empty"
                    rect_color = 'red' if status == "Occupied" else 'green'
                    rect = patches.Rectangle((x, y), 32, 32, linewidth=2, edgecolor=rect_color, facecolor='none')
                    ax[3].add_patch(rect)
                    ax[3].text(x + 1, y + 16, f'{status}\n{conf_score:.2f}', color='white', fontsize=10, ha='left',
                               va='center')
            ax[3].imshow(overlay_image)
            ax[3].set_title('Overlay Image with Confidence')
            ax[3].axis('off')

            plt.show()


# Example call with updated parameters
visualize_with_confidence(model, test_loader, device, num_examples=5, threshold=0.5)

end_time = time.time()
inference_time = end_time - start_time
print(f"Total Time taken for inference: {inference_time:.4f} seconds")