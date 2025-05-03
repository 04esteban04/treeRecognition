import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import json
import os

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout(0.5) 
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        return F.log_softmax(self.fc2(x), dim=1)

def train(model, device, train_loader, optimizer, loss_fn, epochs, test_loader):
    train_losses = []
    test_losses = []
    test_accuracies = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Evaluate on the test set
        test_loss, accuracy = evaluate(model, device, test_loader, loss_fn)
        test_losses.append(test_loss)
        test_accuracies.append(accuracy)

        print(f"Epoch {epoch:3d}: Training Loss = {avg_train_loss:.4f}, "
              f"Test Loss = {test_loss:.4f}, Precision = {accuracy:.2f}%")

    return train_losses, test_losses, test_accuracies

def evaluate(model, device, test_loader, loss_fn):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    return avg_loss, accuracy

def save_metrics_to_file(train_losses, test_losses, test_accuracies, filename="./output/train/trainingMetrics_baseNN.json"):
    metrics = {
        "train_losses": train_losses,
        "test_losses": test_losses,
        "test_accuracies": test_accuracies
    }

    with open(filename, "w") as f:
        json.dump(metrics, f)

    print(f"Metrics saved in '{filename}'")

def plot_metrics_from_file(filename="./output/train/trainingMetrics_baseNN.json"):
    
    # Load metrics from file
    with open(filename, "r") as f:
        metrics = json.load(f)

    train_losses = metrics["train_losses"]
    test_losses = metrics["test_losses"]
    test_accuracies = metrics["test_accuracies"]

    # Plot the metrics
    plt.figure(figsize=(12, 5))

    # Loss metrics
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training')
    plt.plot(test_losses, label='Testing')
    plt.title("Loss vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()

    # Precision metrics
    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies, label='Precision', color='green')
    plt.title("Precision vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Precision (%)")
    plt.grid()

    plt.tight_layout()
    plt.savefig("./output/train/trainingMetrics_baseNN.png")
    plt.show()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.NLLLoss()
    epochs = 10

    train_losses, test_losses, test_accuracies = train(model, device, train_loader, optimizer, loss_fn, epochs, test_loader)

    # Save the model
    os.makedirs("./output/train", exist_ok=True)
    torch.save(model.state_dict(), "./output/train/model_baseNN(MNIST).pth")
    print("Model saved in './output/train/model_baseNN(MNIST).pth'")

    # Save metrics to file and plot
    save_metrics_to_file(train_losses, test_losses, test_accuracies)
    plot_metrics_from_file()

if __name__ == "__main__":
    main()
