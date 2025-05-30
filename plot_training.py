import json
import matplotlib.pyplot as plt


if __name__ == '__main__':
    with open("training_log.json", "r") as f:
        training_log = json.load(f)

    epochs = [entry["epoch"] for entry in training_log]
    losses = [entry["loss"] for entry in training_log]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, marker="o")
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig("training_plot.png")
    plt.show()
