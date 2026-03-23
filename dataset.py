import os
import torch
from torch.utils.data import Dataset
import numpy as np

class LensDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['no', 'sphere', 'vort']
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.image_paths = []
        self.labels = []
        
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.exists(cls_dir):
                continue
            for file_name in os.listdir(cls_dir):
                if file_name.endswith('.npy'):
                    self.image_paths.append(os.path.join(cls_dir, file_name))
                    self.labels.append(self.class_to_idx[cls_name])
                    
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = np.load(img_path)
        
        # Handle normalization if not properly min-max scaled
        if image.max() > 1.0:
            image = (image - image.min()) / (image.max() - image.min() + 1e-8)
            
        # Ensure it's 2D and add channel dim suitable for PyTorch
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=0) # Shape: (1, H, W)
        elif len(image.shape) == 3 and image.shape[-1] in [1, 3]:
            # Convert from (H, W, C) to (C, H, W)
            image = np.transpose(image, (2, 0, 1))
            
        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

if __name__ == "__main__":
    train_dir = '/Users/mywishanand/Documents/1. Multi-Class Classification/dataset/train'
    ds = LensDataset(train_dir)
    print("Total training images:", len(ds))
    if len(ds) > 0:
        img, lbl = ds[0]
        print("Sample shape:", img.shape)
        print("Sample label (int):", lbl.item())
        print("Sample max:", img.max().item(), "min:", img.min().item())
