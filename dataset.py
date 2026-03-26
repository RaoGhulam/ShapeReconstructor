import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, "images")
        self.label_dir = os.path.join(root_dir, "labels")
        self.transform = transform

        self.images = sorted(os.listdir(self.img_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        # Image path
        img_path = os.path.join(self.img_dir, img_name)

        # Label path (replace .png → .txt)
        label_name = img_name.replace(".png", ".txt")
        label_path = os.path.join(self.label_dir, label_name)

        # Load image
        image = Image.open(img_path).convert("L")
        
        if self.transform:
            image = self.transform(image)  # Apply the transform here

        # Load label
        with open(label_path, "r") as f:
            labels = f.readlines()

        shapes = []

        for line in labels:
            parts = line.strip().split(",")
            shape_type = parts[0]

            if shape_type == "circle":
                _, x, y, r = parts
                shapes.append({
                    "shape": "circle",
                    "shape_idx": 0,
                    "params": [float(x), float(y), float(r)]
                })

            elif shape_type == "rectangle":
                _, x1, y1, x2, y2 = parts
                shapes.append({
                    "shape": "rectangle",
                    "shape_idx": 1,
                    "params": [float(x1), float(y1), float(x2), float(y2)]
                })

            elif shape_type == "line":
                _, x1, y1, x2, y2 = parts
                shapes.append({
                    "shape": "line",
                    "shape_idx": 2,
                    "params": [float(x1), float(y1), float(x2), float(y2)]
                })

            elif shape_type == "STOP":
                shapes.append({
                    "shape": "stop",
                    "shape_idx": 3,   # special token
                    "params": []
                })
        
        return image, shapes