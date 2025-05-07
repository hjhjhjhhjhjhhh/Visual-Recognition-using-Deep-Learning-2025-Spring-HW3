import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import cv2
import numpy as np
import tifffile as sio
from pathlib import Path
from torchvision import transforms

class InstanceSegmentationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transforms.Compose([
            transforms.ToTensor()
            # transforms.Normalize(mean=[0.674, 0.539, 0.728], std=[0.177, 0.231, 0.194])
        ])
        self.transform = transform
        self.samples = sorted(self.root_dir.iterdir())

    def __getitem__(self, idx):
        folder = self.samples[idx]
        image = cv2.imread(str(folder / 'image.tif'))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = F.to_tensor(image)

        masks = []
        labels = []
        for class_id in range(1, 5):
            mask_file = folder / f'class{class_id}.tif'
            if mask_file.exists():
                class_mask = sio.imread(mask_file)
                num_labels, label_map = cv2.connectedComponents(class_mask.astype(np.uint8))
                for instance_id in range(1, num_labels):
                    instance_mask = (label_map == instance_id)
                    if np.count_nonzero(instance_mask) == 0:
                        continue
                    masks.append(torch.as_tensor(instance_mask, dtype=torch.uint8))
                    labels.append(class_id)

        boxes = []
        valid_masks = []
        valid_labels = []
        for i, mask in enumerate(masks):
            pos = torch.where(mask)
            if pos[0].numel() == 0 or pos[1].numel() == 0:
                continue

            xmin = torch.min(pos[1])
            xmax = torch.max(pos[1])
            ymin = torch.min(pos[0])
            ymax = torch.max(pos[0])

            if xmax <= xmin or ymax <= ymin:
                continue

            boxes.append([xmin.item(), ymin.item(), xmax.item(), ymax.item()])
            valid_masks.append(mask)
            valid_labels.append(labels[i])

        if len(boxes) == 0:
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "masks": torch.zeros((0, image.shape[1], image.shape[2]), dtype=torch.uint8),
                "image_id": torch.tensor([idx]),
                "area": torch.tensor([]),
                "iscrowd": torch.tensor([]),
            }
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            masks = torch.stack(valid_masks)
            labels = torch.as_tensor(valid_labels, dtype=torch.int64)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

            target = {
                "boxes": boxes,
                "labels": labels,
                "masks": masks,
                "image_id": torch.tensor([idx]),
                "area": area,
                "iscrowd": iscrowd,
            }

        if self.transform:
            image = self.transform(image)

        return image, target

    def __len__(self):
        return len(self.samples)
