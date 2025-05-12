import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import numpy as np
from tabulate import tabulate
import csv
import json
from torchvision.datasets.folder import default_loader

class Net(nn.Module):
    def __init__(self, num_classes, freeze_features=True):
        super(Net, self).__init__()
        self.model = resnet18(weights="IMAGENET1K_V1")
        #self.model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        if freeze_features:
            for param in self.model.parameters():
                param.requires_grad = False

        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)

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

def unnormalize(img_tensor, mean, std):

    for t, m, s in zip(img_tensor, mean, std):
        t.mul_(s).add_(m)

    return img_tensor

def show_predictions_one_by_one(images, preds, targets, probs, class_names, model_accuracy, max_samples=10):
    
    print(f"\nModel global precision: {model_accuracy:.2f}%\n")
    
    plt.ion()
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    for i in range(min(len(images), max_samples)):
        img = unnormalize(images[i].clone(), mean, std)
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)

        pred = preds[i]
        label = targets[i]
        prob = probs[i]

        plt.imshow(img)
        plt.title(f"Predicted: {class_names[pred]} ({prob:.1f}%) | Real value: {class_names[label]}")
        plt.axis('off')
        plt.pause(1)
        plt.clf()
    
    plt.ioff()

def export_prediction_tables(preds, targets, probs, class_names, image_names=None):
    table_all = []
    table_correct = []
    table_incorrect = []

    for i, (pred, target, prob) in enumerate(zip(preds, targets, probs)):
        real_label = class_names[target]
        pred_label = class_names[pred]
        correct = "✔️" if pred == target else "❌"
        row = [i, real_label, pred_label, f"{prob:.1f}", correct]
        table_all.append(row)

        if correct == "✔️":
            table_correct.append(row)
        else:
            table_incorrect.append(row)

    headers = ["Idx", "Real value", "Predicted", "Prob (%)", "Is prediction correct?"]

    # Print table for all predictions
    print("\n" + tabulate(table_all, headers=headers, tablefmt="grid"))

    # Save tables to CSV files
    def save_csv(filename, table):
        with open(filename, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(table)

        print(f"{filename} exported!")

    os.makedirs("./outputModel/test", exist_ok=True)
    save_csv("./outputModel/test/allPredictions_customDataset.csv", table_all)
    save_csv("./outputModel/test/correctPredictions_customDataset.csv", table_correct)
    save_csv("./outputModel/test/wrongPredictions_customDataset.csv", table_incorrect)

def testWithDefaultDataset():
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(f"\nUsing device: {device}")

    data_dir = "../preprocessing/dataset2"
    
    transform = transforms.Compose([
        transforms.Resize((228, 228)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    num_classes = len(test_dataset.classes)
    class_names = test_dataset.classes

    model = Net(num_classes).to(device)
    model.load_state_dict(torch.load("./outputModel/train/model_resnet_best.pth", map_location=device))
    model.eval()

    # Load accuracy from JSON file
    with open("./outputModel/train/trainingMetrics_resnetCustomDataset.json", "r") as f:
        metrics = json.load(f)

    accuracy = metrics.get("model_accuracy", 0.0)

    images, preds, targets, probs = [], [], [], []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            prob_vals = F.softmax(output, dim=1) 
            top_probs, pred = prob_vals.max(dim=1)

            images = data.cpu()
            preds = pred.cpu().tolist()
            targets = target.cpu().tolist()
            probs = (top_probs * 100).cpu().tolist()
            break  # only one batch

    # Use this to show predictions one by one in a window
    #show_predictions_one_by_one(images, preds, targets, probs, class_names, accuracy, max_samples=10)
    
    print(f"\nModel global precision: {accuracy:.2f}%\n")
    export_prediction_tables(preds, targets, probs, class_names)

def guessLabelFromFilename(filename, class_names):
    filename = filename.lower()
    for idx, class_name in enumerate(class_names):
        name_part = class_name.split()[0].lower()
        if name_part in filename:
            return idx
    return -1 

def testWithCustomData(folderPath="../preprocessing/outputPreprocess"):
    device = torch.device("cpu")
    print(f"\nUsing device: {device}")

    transform = transforms.Compose([
        transforms.Resize((228, 228)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    image_paths = [os.path.join(folderPath, f) for f in os.listdir(folderPath)
                   if f.lower().endswith(valid_exts)]

    print(f"Found {len(image_paths)} images in the folder.")
    print(f"Image paths: {image_paths}")

    if not image_paths:
        print("No valid images found in the folder.")
        return

    num_classes = 5  
    class_names = ["arce (Aceraceae)", 
                    "cedro (cedrus deodara)",
                    "eucalipto (eucalyptus globulus)",
                    "pino (pinaceae)",
                    "roble (quercus)"]  

    model = Net(num_classes).to(device)
    model.load_state_dict(torch.load("./outputModel/train/model_resnet_best.pth", map_location=device))
    model.eval()

    with open("./outputModel/train/trainingMetrics_resnetCustomDataset.json", "r") as f:
        metrics = json.load(f)
    accuracy = metrics.get("model_accuracy", 0.0)

    images, preds, targets, probs = [], [], [], []

    for img_path in image_paths:
        try:
            img = default_loader(img_path)  # PIL image
            img_tensor = transform(img).unsqueeze(0).to(device)  # [1, C, H, W]

            with torch.no_grad():
                output = model(img_tensor)
                prob_vals = F.softmax(output, dim=1)
                top_prob, pred = prob_vals.max(dim=1)

            images.append(img_tensor.squeeze(0).cpu())
            preds.append(pred.item())
                        
            filename = os.path.basename(img_path)
            label_idx = guessLabelFromFilename(filename, class_names)
            targets.append(label_idx if label_idx >= 0 else 0)

            probs.append((top_prob.item()) * 100)
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")

    print(f"\n*** Model prediction on custom input images! ***\n")
    print(f"Model global precision (from training): {accuracy:.2f}%\n")
    export_prediction_tables(preds, targets, probs, class_names)

def main():

    testWithDefaultDataset()
    #testWithCustomData()

if __name__ == "__main__":
    main()
