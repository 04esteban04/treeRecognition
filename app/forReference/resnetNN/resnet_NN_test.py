import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)

    def forward(self, x):
        return F.log_softmax(self.model(x), dim=1)

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    return 100. * correct / total

def show_predictions_one_by_one(images, preds, targets, probs, model_accuracy, max_samples=10):
    
    print(f"\nModel global precision: {model_accuracy:.2f}%\n")
    
    plt.ion()
    
    for i in range(min(len(images), max_samples)):
        img = images[i].squeeze(0).numpy()
        pred = preds[i]
        label = targets[i]
        prob = probs[i]

        plt.imshow(img, cmap='gray')
        plt.title(f"Pred: {pred} ({prob:.1f}%) | Real: {label}")
        plt.axis('off')
        plt.pause(1)
        plt.clf()

    plt.ioff()

def main():
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(f"\nUsing device: {device}")

    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    model = Net().to(device)
    model.load_state_dict(torch.load("./output/train/model_resnet(MNIST).pth", map_location=device))
    model.eval()

    accuracy = evaluate(model, test_loader)

    images, preds, targets, probs = [], [], [], []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            prob_vals = output.exp()
            top_probs, pred = prob_vals.max(dim=1)

            images = data.cpu()
            preds = pred.cpu().tolist()
            targets = target.cpu().tolist()
            probs = (top_probs * 100).cpu().tolist()
            break

    show_predictions_one_by_one(images, preds, targets, probs, accuracy, max_samples=10)

if __name__ == "__main__":
    main()
