import json
import torch
from torch.utils.data import DataLoader
from torch import nn, optim

from config import CSV_PATH, BATCH_SIZE, DEVICE, EPOCHS, LR, MODEL_PATH, IMG_DIR, LABEL_MAP_PATH
from dataset_utils import ImageDataset, prepare_dataset
from model_utils import SimpleCNN, get_transforms


def train(train_epochs: int):
    df, label_map = prepare_dataset(CSV_PATH, IMG_DIR, LABEL_MAP_PATH)
    transform = get_transforms()

    train_ds = ImageDataset(df, label_map, img_dir=IMG_DIR, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model = SimpleCNN(num_classes=len(label_map)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    training_log = []
    for epoch in range(train_epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        training_log.append({"epoch": epoch + 1, "loss": avg_loss})
        print(f"Epoch {epoch+1}/{train_epochs}, Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    with open("training_log.json", "w") as f:
        json.dump(training_log, f)


if __name__ == "__main__":
    train(train_epochs=EPOCHS)
