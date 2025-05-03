import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


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

def show_predictions_with_confidence(images, preds, targets, confidences, model_accuracy, n=10):
    
    print(f"\nModel global precision: {model_accuracy:.2f}%\n")
    
    plt.ion()
    
    for i in range(min(n, len(images))):
        image = images[i].squeeze(0).numpy()
        pred = preds[i]
        actual = targets[i]
        confidence = confidences[i]

        plt.imshow(image, cmap='gray')
        plt.title(f"Prediction: {pred} | Real value: {actual}\nConfidence: {confidence:.2f}%")
        plt.axis('off')
        plt.pause(1.0)
        plt.clf()
    plt.ioff()

def main():
    model = Net()
    model.load_state_dict(torch.load("./output/train/model_baseNN(MNIST).pth", map_location=torch.device("cpu")))
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # Get model accuracy
    accuracy = evaluate(model, test_loader)

    # Get a batch of test data
    images, preds, targets, confidences = [], [], [], []
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            probability = torch.exp(output)
            pred = output.argmax(dim=1)
            conf = probability[range(len(pred)), pred] * 100 

            preds.extend(pred.tolist())
            targets.extend(target.tolist())
            confidences.extend(conf.tolist())
            images.extend(data)

            break  # only first batch

    # show predictions with confidence (n samples)
    show_predictions_with_confidence(images, preds, targets, confidences, accuracy, n=10)

if __name__ == "__main__":
    main()
