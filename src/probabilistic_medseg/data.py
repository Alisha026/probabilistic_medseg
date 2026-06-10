from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import torch
from torchvision import transforms
import albumentations as augument
from albumentations.pytorch import ToTensorV2
import os
from PIL import Image

def data_augmentations():
    return augument.Compose([
        augument.HorizontalFlip(),
        augument.VerticalFlip(),
        augument.Rotate(20),
        augument.GaussianBlur(3),
        augument.RandomBrightnessContrast(),
        augument.Normalize(),
        ToTensorV2(),
    ])

def validation_augmentations():
    return augument.Compose([
        augument.Normalize(),
        ToTensorV2(),
    ])

class ISICDataset(Dataset):
    def __init__(self, data_dir, mask_dir, transform=None, num_samples=5):
        
        self.data_dir = data_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.num_samples = num_samples
        

        self.images = sorted([
            i for i in os.listdir(data_dir)
            if i.lower().endswith(('.jpg', ".png"))
        ])
        self.masks = sorted([
            i for i in os.listdir(mask_dir)
            if i.lower().endswith((".jpg", ".png"))
        ])
        

    def __len__(self):
        return len(self.images)
    
    def masks_samples(self, mask):
        # Generate random samples from the mask
        mask_bin =(mask>0.5).astype(np.uint8)
        samples = [mask_bin]
        
        kernel_3= np.ones((3,3),np.uint8)
        kernel_5= np.ones((5,5),np.uint8)
        
       # Add erosion/dilation samples
        if self.num_samples > 1:
            samples.append(cv2.erode(mask_bin, kernel_3, iterations=1))
        if self.num_samples > 2:
            samples.append(cv2.dilate(mask_bin, kernel_3, iterations=1))
        if self.num_samples > 3:
            samples.append(cv2.erode(mask_bin, kernel_5, iterations=1))
        if self.num_samples > 4:
            samples.append(cv2.dilate(mask_bin, kernel_5, iterations=1))
            
        # Ensure we have exactly num_samples
        while len(samples) < self.num_samples:
            samples.append(mask_bin) # Just pad with original if needed
            
        return [s.astype(np.float32) for s in samples[:self.num_samples]]


    def __getitem__(self, index):
        img_name = self.images[index]
        img_basename, img_ext = os.path.splitext(img_name)
        
        # This is the ISIC 2018 Task 1 naming convention
        mask_name = img_basename + '_segmentation.png'

        img_path = os.path.join(self.data_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(img_path).convert("RGB").resize((256, 256))
        image = np.array(image)
        mask = Image.open(mask_path).convert("L").resize((256, 256), Image.Resampling.NEAREST)
        mask = np.array(mask)

        # Generate multiple plausible mask samples
        mask_samples = self.masks_samples(mask)

        if self.transform:
            augmented = self.transform(image=image, masks=mask_samples)
            image = augmented['image']
            masks_list = augmented['masks']
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            masks_list = [torch.from_numpy(m) for m in mask_samples]
        
              
        mask_tensor = torch.stack(masks_list).unsqueeze(1)  # Shape: (num_samples, 1, H, W) and then add channel dim

        return image, mask_tensor
    
    


def dataloader(train_data_dir, train_mask_dir, val_data_dir, val_mask_dir, batch_size, num_workers=4, num_samples=5):
    transform = data_augmentations()
    val_transform = validation_augmentations()

    training_dataset = ISICDataset(data_dir=train_data_dir, mask_dir=train_mask_dir, transform=transform, num_samples=num_samples)
    val_dataset = ISICDataset(data_dir=val_data_dir, mask_dir=val_mask_dir, transform=val_transform, num_samples=num_samples)

    train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader
    