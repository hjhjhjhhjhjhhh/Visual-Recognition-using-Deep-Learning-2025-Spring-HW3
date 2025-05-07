import torch
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import os
import argparse
from preprocess import InstanceSegmentationDataset
from model import build_model

def convert_to_coco_dict(targets, image_ids):
    annotations = []
    images = []

    ann_id = 1
    for img_id, t in zip(image_ids, targets):
        h, w = t["masks"].shape[1:]  # assume all masks same shape

        images.append({
            "id": img_id,
            "height": h,
            "width": w,
        })

        for i in range(len(t["boxes"])):
            # Bounding box in xywh format
            x1, y1, x2, y2 = t["boxes"][i].tolist()
            bbox = [x1, y1, x2 - x1, y2 - y1]
            area = bbox[2] * bbox[3]

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": int(t["labels"][i]),
                "bbox": bbox,
                "area": area,
                "iscrowd": 0,
            })
            ann_id += 1

    return {"images": images, "annotations": annotations, "categories": [
        {"id": i, "name": f"class_{i}"} for i in range(1, 5)  # change if your class set is different
    ]}


def convert_predictions_to_coco(predictions, image_ids):
    results = []
    for img_id, p in zip(image_ids, predictions):
        for i in range(len(p["boxes"])):
            x1, y1, x2, y2 = p["boxes"][i].tolist()
            bbox = [x1, y1, x2 - x1, y2 - y1]
            score = float(p["scores"][i])
            category_id = int(p["labels"][i])
            results.append({
                "image_id": img_id,
                "category_id": category_id,
                "bbox": bbox,
                "score": score,
            })
    return results


def evaluate_map(model, data_loader, device):
    model.eval()
    coco_targets = []
    coco_preds = []
    image_ids = []

    for i, (images, targets) in enumerate(tqdm(data_loader, desc="Evaluating")):
        images = [img.to(device) for img in images]
        with torch.no_grad():
            outputs = model(images)

        image_id = i + 1
        for t in targets:
            t["image_id"] = torch.tensor(image_id)

        for p in outputs:
            p["image_id"] = image_id

        image_ids.append(image_id)
        coco_targets.extend(targets)
        coco_preds.extend(outputs)

    gt_dict = convert_to_coco_dict(coco_targets, image_ids)
    pred_dict = convert_predictions_to_coco(coco_preds, image_ids)

    # Save to temporary JSON files
    os.makedirs("tmp_eval", exist_ok=True)
    with open("tmp_eval/gt.json", "w") as f:
        json.dump(gt_dict, f)
    with open("tmp_eval/pred.json", "w") as f:
        json.dump(pred_dict, f)

    coco_gt = COCO("tmp_eval/gt.json")
    coco_dt = coco_gt.loadRes("tmp_eval/pred.json")

    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval.stats[0]  # mAP@0.5:0.95

def main():
    parser = argparse.ArgumentParser(description="Run inference with Mask R-CNN")
    parser.add_argument('--image-dir', type=str, default='../vrdl_hw3_data', required=True,
                        help='Directory containing test images')
    args = parser.parse_args()

    root = args.image_dir
    dataset = InstanceSegmentationDataset(f'{root}/train')
    val_ratio = 0.1
    val_size = int(val_ratio * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_model('res')
    model.to(device)

    # Optimizer and scheduler
    print(sum(p.numel() for p in model.parameters()))

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-7) 
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, min_lr=1e-8, mode='min')

    # Train loop
    num_epochs = 20
    best_map = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=False)
        for images, targets in loop:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            running_loss += losses.item()
            loop.set_postfix(loss=losses.item())

        print(f"Epoch {epoch+1} - Train Loss: {(running_loss / len(train_loader)):.4f}")

        val_map = evaluate_map(model, val_loader, device)
        print(f"Epoch {epoch+1} - Validation mAP: {val_map:.4f}")

        
        # Save best model
        if val_map > best_map:
            best_map = val_map
            torch.save(model.state_dict(), 'best_model.pth')
            print("Saved new best model.")
        lr_scheduler.step(val_map)
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()

