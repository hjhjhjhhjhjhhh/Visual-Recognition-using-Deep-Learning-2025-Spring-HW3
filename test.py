import torch
import json
from pathlib import Path
import cv2
import argparse
import torchvision.transforms.functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import maskrcnn_resnet50_fpn
from tqdm import tqdm
from utils import encode_mask


# Load model
def load_model(model_type, weight_path, device):
    model = maskrcnn_resnet50_fpn(pretrained=False)
    num_classes = 5  # 4 classes + background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        model.roi_heads.mask_predictor.conv5_mask.in_channels,
        256,
        num_classes
    )
    model.eval()
    model.to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    return model

# Prediction function
def predict(model, json_input_path, image_dir, output_json_path, score_threshold=0.5, device='cuda'):
    with open(json_input_path, 'r') as f:
        image_entries = json.load(f)

    results = []
    model.eval()

    for entry in tqdm(image_entries):
        image_path = Path(image_dir) / entry["file_name"]
        image_id = entry["id"]
        height = entry["height"]
        width = entry["width"]

        # Load and preprocess image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = F.to_tensor(image).to(device)

        # Model expects a list of images
        with torch.no_grad():
            outputs = model([image_tensor])[0]

        for i in range(len(outputs['scores'])):
            score = outputs['scores'][i].item()
            if score < score_threshold:
                continue

            box = outputs['boxes'][i].tolist()  # [xmin, ymin, xmax, ymax]
            # Convert to [x, y, w, h]
            x, y, xmax, ymax = box
            w = xmax - x
            h = ymax - y

            label = outputs['labels'][i].item()
            mask = outputs['masks'][i, 0].cpu().numpy() > 0.5  # binary mask

            rle = encode_mask(mask)

            result = {
                "image_id": image_id,
                "bbox": [x, y, w, h],
                "score": score,
                "category_id": label,
                "segmentation": {
                    "size": [height, width],
                    "counts": rle["counts"]
                }
            }
            results.append(result)

    # Write results to output JSON
    with open(output_json_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Saved {len(results)} predictions to {output_json_path}")

def main():
    parser = argparse.ArgumentParser(description="Run inference with Mask R-CNN")
    parser.add_argument('--model-type', type=str, choices=['res', 'swin_s'], default='res',
                        help='Model backbone type: "res" or "swin_s"')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to model weights (.pth file)')
    parser.add_argument('--image-dir', type=str, default='../vrdl_hw3_data', required=True,
                        help='Directory containing test images')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.model_type, args.weights, device)

    root = '../vrdl_hw3_data'
    json_input_path = f'{root}/test_image_name_to_ids.json'
    test_folder = f'{root}/test_release'
    predict(
        model=model,
        json_input_path=json_input_path,
        image_dir=test_folder,
        output_json_path='test-results.json',
        score_threshold=0.5,
        device=device
    )

if __name__ == '__main__':
    main()
