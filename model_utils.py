import torch
import torch.nn as nn
from torchvision import transforms
import json

from config import IMAGE_SIZE, DEVICE, MODEL_PATH, LABEL_MAP_PATH


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 25 * 25, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def get_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])


def load_model() -> tuple[nn.Module, dict[int, str]]:
    with open(LABEL_MAP_PATH, 'r') as f:
        label_map = json.load(f)
    inv_label_map = {v: k for k, v in label_map.items()}
    model = SimpleCNN(num_classes=len(label_map))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    model.to(DEVICE)
    return model, inv_label_map
