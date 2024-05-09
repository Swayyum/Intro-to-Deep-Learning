#%%
import torch

if torch.cuda.is_available():
    n_gpu = torch.cuda.device_count()
    print(f"Number of GPUs available: {n_gpu}")
    for i in range(n_gpu):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No GPU is available.")
#%%
import os

# Define the path to the dataset and the partitions
# dataset_path = 'Data'
dataset_path = r'C:/Users/SirM/Desktop/Swayam/Intro to Deep Learning/Intro-to-Deep-Learning/Final Project/PKLot.v1-raw.yolov8-obb'
partition = 'train'
images_folder = 'images'
labels_folder = 'labels'

# Construct the path to the images folder within the train partition
images_path = os.path.join(dataset_path, partition, images_folder)
labels_folder_path = os.path.join(dataset_path, partition, labels_folder)

# Count the number of files in the images folder
num_files = len([name for name in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, name))])
num_text_files = len([name for name in os.listdir(labels_folder_path) if os.path.isfile(os.path.join(labels_folder_path, name))])

print(f"Number of files in folder 'images' in 'train': {num_files}")
print(f"Number of files in folder 'labels' in 'train': {num_text_files}")

#%%
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms
#import matplotlib.patches as patches

def parse_annotation(annotation_line):
    parts = annotation_line.strip().split()
    class_id = int(parts[0])  # The class_id is the first element
    vertices = np.array(parts[1:], dtype=np.float32)  # The rest are the vertices
    return class_id, vertices.reshape((-1, 2))  # Reshape to Nx2 where N is the number of vertices

def draw_polygon_on_mask(mask, corners, image_shape):
    scaled_corners = corners * np.array([image_shape[1], image_shape[0]], dtype=np.float32)  # scale x and y
    scaled_corners = np.around(scaled_corners).astype(np.int32)  # round and convert to int

    # print("Scaled Corners:", scaled_corners)  # Debugging print

    corners_int = scaled_corners.reshape((-1, 1, 2))
    cv2.fillPoly(mask, [corners_int], color=(255))  # Ensure fillPoly is used, not polylines

    # # Debugging visualization
    # plt.imshow(mask, cmap='gray')
    # plt.title('Polygon on Mask')
    # plt.axis('off')
    # plt.show()

def create_mask_from_annotations(annotation_path, image_shape):
    mask = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)  # Create a black mask
    with open(annotation_path, 'r') as file:
        for line in file:
            class_id, vertices = parse_annotation(line)
            draw_polygon_on_mask(mask, vertices, image_shape)  # Draw each polygon on the mask
    return mask

def preprocess_image(image_path, annotation_path, target_size):
    # Open and resize image
    image = Image.open(image_path).resize(target_size)
    mask = create_mask_from_annotations(annotation_path, target_size)  # Note the reversal of width and height for the mask
    return np.array(image), mask

# def debug_visualization(image, mask, corners, title='Debug Visualization'):
#     fig, ax = plt.subplots(1, 2, figsize=(10, 5))
#
#     # Show original image with annotations
#     ax[0].imshow(image)
#     ax[0].set_title(f'{title}: Original Image with Annotations')
#     for corner in corners:
#         scaled_corners = corner * np.array([image.shape[1], image.shape[0]])
#         polygon = patches.Polygon(scaled_corners, linewidth=1, edgecolor='r', facecolor='none')
#         ax[0].add_patch(polygon)
#
#     # Show mask
#     ax[1].imshow(mask, cmap='gray')
#     ax[1].set_title(f'{title}: Mask')
#
#     plt.tight_layout()
#     plt.show()

def process_directory(data_dir, annotation_dir, target_size):
    for img_filename in os.listdir(data_dir):
        if img_filename.endswith('.jpg'):
            # Paths for the image and its corresponding annotation file
            image_path = os.path.join(data_dir, img_filename)
            annotation_path = os.path.join(annotation_dir, img_filename.replace('.jpg', '.txt'))

            # Check if the annotation file exists
            if not os.path.isfile(annotation_path):
                print(f"Annotation file does not exist for {image_path}")
                continue  # Skip this image

            # Preprocess the image and create the mask from the annotation
            image, mask = preprocess_image(image_path, annotation_path, target_size)

            # Visualization for debugging
            corners = []  # Store the vertices for debugging visualization
            with open(annotation_path, 'r') as file:
                for line in file:
                    class_id, vertices = parse_annotation(line)
                    corners.append(vertices)

            #debug_visualization(image, mask, corners, title=os.path.basename(image_path))

            # Save the processed image and mask as .npy files
            image_npy_path = os.path.join(data_dir, img_filename.replace('.jpg', '_img.npy'))
            mask_npy_path = os.path.join(data_dir, img_filename.replace('.jpg', '_mask.npy'))

            np.save(image_npy_path, image)
            np.save(mask_npy_path, mask)

def save_as_npy(data_dir, annotation_dir, target_size=(224, 224)):
    processed_data = []
    for img_filename in os.listdir(data_dir):
        if img_filename.endswith('.jpg'):
            image_path = os.path.join(data_dir, img_filename)
            annotation_path = os.path.join(annotation_dir, img_filename.replace('.jpg', '.txt'))
            image, mask = preprocess_image(image_path, annotation_path, target_size)
            np.save(os.path.join(data_dir, img_filename.replace('.jpg', '_img.npy')), image)
            np.save(os.path.join(data_dir, img_filename.replace('.jpg', '_mask.npy')), mask)
            processed_data.append((image, mask))
    print(f"Processed and saved {len(processed_data)} image-mask pairs in .npy format.")

# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.imshow(image)
# plt.title('Processed Image')
# plt.axis('off')
#
# plt.subplot(1, 2, 2)
# plt.imshow(mask, cmap='gray')
# plt.title('Processed Mask')
# plt.axis('off')
#
# plt.show()
# Example usage
dataset_base = r'C:/Users/SirM/Desktop/Swayam/Intro to Deep Learning/Intro-to-Deep-Learning/Final Project/PKLot.v1-raw.yolov8-obb'
#dataset_base = r'Data'
partitions = ['train', 'valid', 'test']
target_size = (512,512)  # Change as required by your model

# Process images and annotations and save them as .npy files
# Process images and annotations
for part in partitions:
    images_dir = os.path.join(dataset_base, part, 'images')
    annotations_dir = os.path.join(dataset_base, part, 'labels')
    process_directory(images_dir, annotations_dir, target_size)
    print(f"Finished processing {part} set")

# Visualization (example for one image from the 'train' set)
train_images_dir = os.path.join(dataset_base, 'train', 'images')
train_image_files = [f for f in os.listdir(train_images_dir) if f.endswith('_img.npy')]

# Load one image and its corresponding mask
image = np.load(os.path.join(train_images_dir, train_image_files[0]))
mask = np.load(os.path.join(train_images_dir, train_image_files[0].replace('_img.npy', '_mask.npy')))

# # Visualize the image and the mask
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.imshow(image)
# plt.title('Processed Image')
# plt.axis('off')
#
# plt.subplot(1, 2, 2)
# plt.imshow(mask, cmap='gray')
# plt.title('Processed Mask')
# plt.axis('off')
#
# plt.show()