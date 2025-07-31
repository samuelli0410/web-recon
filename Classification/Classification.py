import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.io import readsav
import sys
from pathlib import Path
import csv
from PIL import Image
from torch.utils.data import random_split
import pandas as pd


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            #input is 3*224*224
            nn.Conv2d(3, 64, 7), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(64, 128, 6), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            
            nn.AdaptiveAvgPool2d((2, 2))
        )
        self.classifier =nn.Linear(256*4, 15)
    def forward(self,x):
        x = self.features(x)
        x=x.view(x.size(0),-1)
        x=self.classifier(x)
        return x


def train_classifier(training_data, batch_size=16, num_epochs=50,learning_rate = 1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"training on {device}")

    model = Classifier().to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)


    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle= True)
    # val_loader = DataLoader(validation_data, batch_size=batch_size, shuffle= True)
    

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            optimizer.step()
            running_loss += loss.item()
        model.eval()


        # val_loss = 0.0
        # with torch.no_grad():
        #     for inputs, targets in val_loader:
        #         inputs, targets = inputs.to(device), targets.to(device)
        #         outputs = model(inputs)
        #         loss = criterion(outputs, targets)
        #         val_loss += loss.item()


        # avg_val_loss = val_loss / len(val_loader)
        # print(f"avg val loss: {avg_val_loss}")


    return model




class SpiderDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # Converts to [0, 1] and (C, H, W)
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)  # Normalize RGB
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]["image_path"]
        label = int(self.data.iloc[idx]["label"])
        image = Image.open(img_path).convert("RGB")  # Ensure 3 channels

        image = self.transform(image)
        return image, label
    

path = "C:/Users/samue/Downloads/archive/spiders.csv"

data = SpiderDataset(path)

train_classifier(data)

