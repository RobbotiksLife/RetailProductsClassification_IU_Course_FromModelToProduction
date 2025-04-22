import torch

IMG_DIR = "retail-products-classification/train"
CSV_PATH = "retail-products-classification/train.csv"
MODEL_PATH = "model.pth"
LABEL_MAP_PATH = "label_map.json"
IMAGE_SIZE = 100
BATCH_SIZE = 32
EPOCHS = 25
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
