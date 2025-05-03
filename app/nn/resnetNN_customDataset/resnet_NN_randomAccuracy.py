import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import json
import os
from sklearn.metrics import classification_report, confusion_matrix

class Net(nn.Module):
    def __init__(self, num_classes, freeze_features=True):
        super(Net, self).__init__()
        self.model = resnet18(weights="IMAGENET1K_V1")

        if freeze_features:
            for param in self.model.parameters():
                param.requires_grad = False

        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)

def train(model, device, train_loader, optimizer, loss_fn, epochs, test_loader, scheduler=None):
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

        # only use scheduler.step() if StepLR is used
        # only use scheduler.step(test_loss) if ReduceLROnPlateau is used
        if scheduler:
            scheduler.step()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        test_loss, accuracy = evaluate(model, device, test_loader, loss_fn)
        test_losses.append(test_loss)
        test_accuracies.append(accuracy)

        #if scheduler:
        #    scheduler.step(test_loss)

        print(f"Epoch {epoch:2d}: Training Loss = {avg_train_loss:.4f}, Test Loss = {test_loss:.4f}, Precision = {accuracy:.2f}%")

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

def evaluate_detailed(model, loader, device, class_names):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            preds = output.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    print("\nDetailed clasification:")
    print(classification_report(all_targets, all_preds, target_names=class_names, digits=4))
    
    print("\nConfusion matrix:")
    print(confusion_matrix(all_targets, all_preds))

def save_metrics_to_file(train_losses, test_losses, test_accuracies, filename="trainingMetrics_resnetRandomAccuracy.json"):
    
    metrics = {
        "train_losses": train_losses,
        "test_losses": test_losses,
        "test_accuracies": test_accuracies
    }

    with open(filename, "w") as f:
        json.dump(metrics, f)

    print(f"Metrics saved in '{filename}'")

def plot_metrics_from_file(filename="trainingMetrics_resnetRandomAccuracy.json"):
    with open(filename, "r") as f:
        metrics = json.load(f)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(metrics["train_losses"], label='Entrenamiento')
    plt.plot(metrics["test_losses"], label='Prueba')
    plt.title("Loss vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(metrics["test_accuracies"], label='Precision (%)', color='green')
    plt.title("Precision vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Precision (%)")
    plt.grid()

    plt.tight_layout()
    plt.savefig("trainingMetrics_resnetRandomAccuracy.png")
    plt.show()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dir = "./dataset2"

    # Aumentation for training
    train_transform = transforms.Compose([
        transforms.Resize((228, 228)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((228, 228)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_transform)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    num_classes = len(train_dataset.classes)
    model = Net(num_classes=num_classes, freeze_features=False).to(device)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    epochs = 20

    train_losses, test_losses, test_accuracies = train(model, device, train_loader, optimizer, loss_fn, epochs, test_loader, scheduler)

    torch.save(model.state_dict(), "model_resnet(randomAccuracy).pth")
    print("Model saved as 'model_resnet(randomAccuracy).pth'")

    save_metrics_to_file(train_losses, test_losses, test_accuracies)
    plot_metrics_from_file()

    # Detailed evaluation with classification report and confusion matrix
    evaluate_detailed(model, test_loader, device, train_dataset.classes)

if __name__ == "__main__":
    main()
