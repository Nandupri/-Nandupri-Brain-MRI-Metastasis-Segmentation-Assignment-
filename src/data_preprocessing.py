import os
import cv2
import numpy as np
from skimage import exposure
from torchvision import transforms

def clahe_preprocess(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image_clahe = clahe.apply(image)
    return image_clahe

def normalize(image):
    return image / 255.0

def augment(image, mask):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.ToTensor()
    ])
    image = transform(image)
    mask = transform(mask)
    return image, mask

def preprocess_data(raw_images_dir, raw_masks_dir, processed_images_dir, processed_masks_dir):
    os.makedirs(processed_images_dir, exist_ok=True)
    os.makedirs(processed_masks_dir, exist_ok=True)
    
    for img_name in os.listdir(raw_images_dir):
        img_path = os.path.join(raw_images_dir, img_name)
        mask_path = os.path.join(raw_masks_dir, img_name)  # Assuming mask has same name
        
        if not os.path.exists(mask_path):
            continue  # Skip if mask is missing
        
        image = clahe_preprocess(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None or mask is None:
            continue  # Skip if any image is corrupted
        
        image = normalize(image)
        mask = normalize(mask)
        
        # Save preprocessed images
        cv2.imwrite(os.path.join(processed_images_dir, img_name), image * 255)
        cv2.imwrite(os.path.join(processed_masks_dir, img_name), mask * 255)
        
        # Apply augmentations
        aug_image, aug_mask = augment(image, mask)
        aug_image = aug_image.numpy().transpose(1, 2, 0) * 255
        aug_mask = aug_mask.numpy().transpose(1, 2, 0) * 255
        
        aug_img_name = f"aug_{img_name}"
        cv2.imwrite(os.path.join(processed_images_dir, aug_img_name), aug_image)
        cv2.imwrite(os.path.join(processed_masks_dir, aug_img_name), aug_mask)
