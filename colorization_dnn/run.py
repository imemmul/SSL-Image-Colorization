import torch
from model import Colorization_Model
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from torchsummary import summary

dataset_ab = '/home/emir/Desktop/dev/myResearch/dataset/colorization_lab.npy'
dataset_gray = '/home/emir/Desktop/dev/myResearch/dataset/l/gray_scale.npy'

device = "cuda"

def train(model, train_loader, val_loader, epochs):
    print("Training starting")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                
        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

    torch.save(model.state_dict(), 'colorization_weights.pth')

def run():
    
    gray = np.load(dataset_gray)
    ab = np.load(dataset_ab)
    train_in = gray[:1500]
    train_out = ab[:1500]
    
    train_in = np.repeat(train_in[..., np.newaxis], 3, -1)
    x = torch.FloatTensor(train_in.transpose(0, 3, 1, 2)).to(device)  # NHWC to NCHW
    y = torch.FloatTensor(train_out.transpose(0, 3, 1, 2)).to(device)
    dataset = TensorDataset(x, y)
    # print(x.shape)
    # print(y.shape)
    model = Colorization_Model(in_channels=3)
    
    batch_size = 8
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    model.to(device)  # Send model to GPU if available
    summary(model, (3,224,224))
    # train(model=model, train_loader=train_loader, val_loader=val_loader, epochs=50)

# Assuming you have set the device
run()
