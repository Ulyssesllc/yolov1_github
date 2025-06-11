# YOLOv8nCustom Vehicle Detection

This project implements a fast, minimal YOLOv8n-style object detection model from scratch in PyTorch, designed for vehicle detection tasks. The model architecture is lightweight and suitable for real-time applications or experimentation.

## Project Structure

```
model_architecture/
    architecture.py      # YOLOv8nCustom model definition
vehicle_dataset4/
    data.yaml            # Dataset configuration for training/validation
    train/
        images/          # Training images
        labels/          # Training labels (YOLO format)
    valid/
        images/          # Validation images
        labels/          # Validation labels (YOLO format)
    test/
        images/          # Test images
        labels/          # Test labels (if available)
yolov8n_custom.yaml      # (Optional) Model or training configuration
```

## Model Overview

- **Backbone:** Stacked Conv and C2f blocks for feature extraction.
- **Neck:** SPPF block for multi-scale context aggregation.
- **Head:** Single detection head for fast inference (predicts bounding boxes, objectness, and class scores).
- **Framework:** PyTorch

## Usage

### 1. Model Inference Example

```python
from model_architecture.architecture import YOLOv8nCustom
import torch

model = YOLOv8nCustom(num_classes=12)
dummy = torch.randn(1, 3, 640, 640)
out = model(dummy)
print(out.shape)  # Output: [1, (num_classes+5)*3, H, W]
```

### 2. Training

- Prepare your dataset in YOLO format (see `vehicle_dataset4/`).
- Implement a training loop with a YOLO-style loss function.
- Use the model defined in `architecture.py`.

### 3. Dataset

- Images and labels are organized in `train/`, `valid/`, and `test/` folders.
- Labels follow the YOLO format: `<class> <x_center> <y_center> <width> <height>` (normalized).

## Notes

- The model is designed for speed and simplicity, making it suitable for real-time or resource-constrained environments.
- For best results, tune the model architecture, anchors, and training hyperparameters to your specific dataset.
- This implementation is original and does not copy code from any proprietary or licensed sources.

## References

- [YOLOv8 Paper (Ultralytics)](https://docs.ultralytics.com/models/yolov8/)
- [YOLOv5/YOLOv8 Documentation](https://docs.ultralytics.com/)
- [PyTorch Documentation](https://pytorch.org/)

---


