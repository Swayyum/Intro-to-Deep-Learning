# %%
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import seaborn as sns
import time
from matplotlib.colors import Normalize
from torchvision.transforms.functional import to_tensor, resize
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RescaleTransform:
    """Transform that rescales the input NumPy arrays to the given 'output_size'."""

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image, mask):
        # Convert NumPy arrays to tensors
        image = to_tensor(image)
        mask = to_tensor(mask)
        # Resize images and masks
        image = resize(image, self.output_size)
        mask = resize(mask, self.output_size)
        return image, mask


class AlexNetSegmentation(nn.Module):
    def __init__(self):
        super(AlexNetSegmentation, self).__init__()
        alexnet = models.alexnet(pretrained=True).features
        self.encoder = nn.Sequential(*list(alexnet.children())[:6])  # Adjust as needed

        # Assuming the last layer output channels
        out_channels = 192  # This should match the actual output of your encoder's last Conv layer
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(out_channels, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        encoder_output = x
        x = self.decoder(x)
        x = TF.center_crop(x, [256, 256])
        return x


class PKLotDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith('_img.npy')]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = os.path.join(self.data_dir, self.image_files[idx])
        mask_file = image_file.replace('_img.npy', '_mask.npy')
        image = np.load(image_file)
        mask = np.load(mask_file)
        if self.transform:
            image, mask = self.transform(image, mask)
        return image, mask


dataset_base = r'C:/Users/SirM/Desktop/Swayam/Intro to Deep Learning/Intro-to-Deep-Learning/Final Project/PKLot.v1-raw.yolov8-obb'
partitions = ['train', 'valid', 'test']
transform = RescaleTransform(output_size=(256, 256))

train_dataset = PKLotDataset(os.path.join(dataset_base, 'train', 'images'), transform=transform)
valid_dataset = PKLotDataset(os.path.join(dataset_base, 'valid', 'images'), transform=transform)
test_dataset = PKLotDataset(os.path.join(dataset_base, 'test', 'images'), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

model = AlexNetSegmentation().to(device)
summary(model, input_size=(3, 256, 256))

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()


# Training and evaluation functions
def train_one_epoch(model, dataloader, device, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct_pixels = 0
    total_pixels = 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

        # Calculate accuracy
        predicted = outputs > 0.5
        correct_pixels += (predicted == labels).sum().item()
        total_pixels += torch.numel(labels)

    epoch_loss = running_loss / len(dataloader.dataset)
    accuracy = correct_pixels / total_pixels
    return epoch_loss, accuracy


def validate(model, dataloader, device, criterion):
    model.eval()
    running_loss = 0.0
    correct_pixels = 0
    total_pixels = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            # Calculate accuracy
            predicted = outputs > 0.5
            correct_pixels += (predicted == labels).sum().item()
            total_pixels += torch.numel(labels)

    epoch_loss = running_loss / len(dataloader.dataset)
    accuracy = correct_pixels / total_pixels
    return epoch_loss, accuracy


train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

for epoch in range(10):  # Example number of epochs
    train_loss, train_accuracy = train_one_epoch(model, train_loader, device, optimizer, criterion)
    val_loss, val_accuracy = validate(model, valid_loader, device, criterion)

    # Append metrics to the lists
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    print(
        f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')


def visualize_with_confidence(model, loader, device, num_examples=5, threshold=0.5):
    model.eval()
    indices = torch.randperm(len(loader.dataset))[:num_examples]  # Randomly select indices
    subset = torch.utils.data.Subset(loader.dataset, indices)  # Create a subset based on these indices
    sub_loader = torch.utils.data.DataLoader(subset, batch_size=1)  # Load the subset

    cmap = plt.get_cmap('coolwarm')
    norm = Normalize(vmin=0, vmax=1)

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

            fig, ax = plt.subplots(1, 6, figsize=(30, 5))  # Add another subplot for confidence score text

            ax[0].imshow(images_np[0].transpose(1, 2, 0))
            ax[0].set_title('Original Image')
            ax[0].axis('off')

            ax[1].imshow(masks_np[0].squeeze(), cmap='gray')
            ax[1].set_title('True Mask')
            ax[1].axis('off')

            ax[2].imshow(preds_np[0].squeeze(), cmap='gray')
            ax[2].set_title('Predicted Mask')
            ax[2].axis('off')

            confidence_overlay = cmap(
                norm(sigmoid_outputs_np[0].squeeze()))  # Use the normalized confidence values for coloring
            ax[3].imshow(images_np[0].transpose(1, 2, 0))
            ax[3].imshow(confidence_overlay, alpha=0.6)  # Overlay confidence heatmap with transparency
            ax[3].set_title('Confidence Heatmap')
            ax[3].axis('off')

            # Confidence histogram
            confidence_values = sigmoid_outputs_np[0].squeeze().flatten()
            ax[4].hist(confidence_values, bins=20, color='blue', alpha=0.7)
            ax[4].set_title('Confidence Histogram')
            ax[4].set_xlabel('Confidence Score')
            ax[4].set_ylabel('Frequency')


# Example call with updated parameters
visualize_with_confidence(model, test_loader, device, num_examples=5, threshold=0.5)

fig, ax = plt.subplots(2, 1, figsize=(10, 10))

# Plot training and validation loss
ax[0].plot(train_losses, label='Training Loss')
ax[0].plot(val_losses, label='Validation Loss')
ax[0].set_title('Training and Validation Loss')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss')
ax[0].legend()

# Plot training and validation accuracy
ax[1].plot(train_accuracies, label='Training Accuracy')
ax[1].plot(val_accuracies, label='Validation Accuracy')
ax[1].set_title('Training and Validation Accuracy')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Accuracy')
ax[1].legend()

# Display the plot
plt.tight_layout()
plt.show()


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

