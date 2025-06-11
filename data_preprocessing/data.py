import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader


class YOLODataset(Dataset):
    def __init__(self, images_dir, labels_dir, img_size=640, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.img_size = img_size
        self.transform = transform
        self.image_files = [
            f for f in os.listdir(images_dir) if f.endswith((".jpg", ".png", ".jpeg"))
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        label_path = os.path.join(
            self.labels_dir, os.path.splitext(img_name)[0] + ".txt"
        )

        # Load image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img / 255.0  # Normalize to [0, 1]
        img = torch.from_numpy(img).permute(2, 0, 1).float()

        # Load labels (YOLO format: class x_center y_center width height)
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    class_id, x, y, w, h = map(float, line.strip().split())
                    boxes.append([class_id, x, y, w, h])
        boxes = (
            torch.tensor(boxes, dtype=torch.float32)
            if boxes
            else torch.zeros((0, 5), dtype=torch.float32)
        )

        if self.transform:
            img = self.transform(img)

        return img, boxes


# Example usage:
if __name__ == "__main__":
    images_dir = "../vehicle_dataset4/train/images"
    labels_dir = "../vehicle_dataset4/train/labels"
    dataset = YOLODataset(images_dir, labels_dir)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    for imgs, targets in dataloader:
        print(imgs.shape, targets.shape)
        break
