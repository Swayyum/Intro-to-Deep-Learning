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
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class ResNetUNet(nn.Module):
    def __init__(self):
        super(ResNetUNet, self).__init__()
        # Encoder
        self.init_conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.init_bn = nn.BatchNorm2d(64)
        self.init_relu = nn.ReLU(inplace=True)
        self.init_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet blocks
        self.encoder1 = ResidualBlock(64, 64)
        self.encoder2 = ResidualBlock(64, 128, stride=2, downsample=nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(128)
        ))
        self.encoder3 = ResidualBlock(128, 256, stride=2, downsample=nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(256)
        ))
        self.encoder4 = ResidualBlock(256, 512, stride=2, downsample=nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(512)
        ))

        # Decoders
        self.decoder4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)

        # Final layer
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.init_conv(x)
        x = self.init_bn(x)
        x = self.init_relu(x)
        x = self.init_pool(x)

        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        dec4 = self.decoder4(enc4) + enc3
        dec3 = self.decoder3(dec4) + enc2
        dec2 = self.decoder2(dec3) + enc1
        dec1 = self.decoder1(dec2)

        out = self.final_conv(dec1)
        out = self.sigmoid(out)
        out = F.interpolate(out, size=(512, 512), mode='bilinear', align_corners=False)
        return out


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


# print(outputs.shape, masks.shape)
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        y_pred_flat = y_pred.view(-1)
        y_true_flat = y_true.view(-1)
        intersection = (y_pred_flat * y_true_flat).sum()
        dice_coeff = (2. * intersection + self.smooth) / (y_pred_flat.sum() + y_true_flat.sum() + self.smooth)
        return 1 - dice_coeff


# print("Output shape:", outputs.shape)  # Expected shape: [batch, 1, H, W]
# print("Mask shape:", masks.shape)     # Expected shape: [batch, 1, H, W]
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
model = ResNetUNet().to(device)
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


# for epoch in range(1):  # Example with 1 epoch for brevity
#     model.train()
#     for images, masks in train_loader:
#         images, masks = images.to(device), masks.to(device)
#         outputs = model(images)
#
#         # Now 'outputs' and 'masks' are defined and we can print their shapes here
#         print("Output shape:", outputs.shape)  # Expected shape: [batch, 1, H, W]
#         print("Mask shape:", masks.shape)     # Expected shape: [batch, 1, H, W]
#
#         # Compute loss
#         loss = criterion(outputs, masks)
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()
#
#         # Optionally break after one iteration for initial debugging
#         break
#
#     # After exiting the loop, 'outputs' and 'masks' are not defined here anymore
#
#     # Print training progress (loss, accuracy, etc.)
#     print(f"Epoch {epoch+1}: Loss: {loss.item()}")
#
# # If you need to check the last batch's shapes outside the loop, do it within the loop and store results
# last_output_shape = outputs.shape
# last_mask_shape = masks.shape
# print("Last output shape recorded inside loop:", last_output_shape)
# print("Last mask shape recorded inside loop:", last_mask_shape)
#
# Training loop
for epoch in range(10):  # Running only for 1 epoch for testing, adjust as needed
    model.train()
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)

        # Print shapes here where outputs and masks are defined
        # print("Output shape:", outputs.shape)  # Expected shape: [batch, 1, H, W]
        # print("Mask shape:", masks.shape)     # Expected shape: [batch, 1, H, W]

        # Compute loss
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

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


def check_gradients(model):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f'{name} gradient norm: {param.grad.norm()}')


def check_predictions(model, loader, device):
    model.eval()
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            sigmoid_outputs = torch.sigmoid(outputs)
            print("Sigmoid outputs:", sigmoid_outputs)
            break  # Check just one batch for quick inspection


check_predictions(model, test_loader, device)


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
            grid_size = 64  # Adjust this value as needed to make squares larger or smaller
            for y in range(0, preds_np[0].shape[1], grid_size):
                for x in range(0, preds_np[0].shape[2], grid_size):
                    conf_score = np.mean(sigmoid_outputs_np[0][:, y:y + grid_size, x:x + grid_size])
                    status = "Occupied" if conf_score > threshold else "Empty"
                    rect_color = 'red' if status == "Occupied" else 'green'
                    rect = patches.Rectangle((x, y), grid_size, grid_size, linewidth=2, edgecolor=rect_color,
                                             facecolor='none')
                    ax[3].add_patch(rect)
                    ax[3].text(x + grid_size / 2, y + grid_size / 2, f'{status}\n{conf_score:.2f}', color='white',
                               fontsize=10, ha='center', va='center')
            ax[3].imshow(overlay_image)
            ax[3].set_title('Overlay Image with Confidence')
            ax[3].axis('off')

            plt.show()


# Example call with updated parameters
visualize_with_confidence(model, test_loader, device, num_examples=5, threshold=0.5)

end_time = time.time()
inference_time = end_time - start_time
print(f"Total Time taken for inference: {inference_time:.4f} seconds")