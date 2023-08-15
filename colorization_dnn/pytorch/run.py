import torch
from model import Colorization_Model
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from torchsummary import summary
import argparse
dataset_ab = '/home/emir/Desktop/dev/myResearch/dataset/colorization_lab.npy'
dataset_gray = '/home/emir/Desktop/dev/myResearch/dataset/l/gray_scale.npy'

device = "cuda"


parser = argparse.ArgumentParser(description='Colorization')
parser.add_argument('--resume', action='store_true', help='Resume training from saved model')
parser.add_argument('--epochs', type=int, default=50, help='Number of additional epochs to train (default: 50)')
parser.add_argument('--weights', type=str, default='', help='Path to save model weights')
args = parser.parse_args()

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

    torch.save(model.state_dict(), './colorization_weights.pth')

def run():
    # TODO 1500 quite low make it 5000 at least
    # TODO increase batch size to 16 - 32
    # TODO you changed in_channels=1 for grayscale so you need to fix model for that output should be 2x224x224
    gray = np.load(dataset_gray)
    ab = np.load(dataset_ab)
    train_in = gray[:5000]
    train_out = ab[:5000]
    
    train_in = np.repeat(train_in[..., np.newaxis], 3, -1)
    x = torch.FloatTensor(train_in.transpose(0, 3, 1, 2)).to(device)  # NHWC to NCHW
    y = torch.FloatTensor(train_out.transpose(0, 3, 1, 2)).to(device)
    dataset = TensorDataset(x, y)
    # print(x.shape)
    # print(y.shape)
    model = Colorization_Model(in_channels=3)
    print(x.shape)
    print(y.shape)
    batch_size = 8
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    model.to(device)
    summary(model, (3,224,224))
    train(model=model, train_loader=train_loader, val_loader=val_loader, epochs=50)


def load_dataset():
    gray = np.load(dataset_gray)
    ab = np.load(dataset_ab)
    train_in = gray[:5000]
    train_out = ab[:5000]
    
    train_in = np.repeat(train_in[..., np.newaxis], 3, -1)
    x = torch.FloatTensor(train_in.transpose(0, 3, 1, 2)).to(device)  # NHWC to NCHW
    y = torch.FloatTensor(train_out.transpose(0, 3, 1, 2)).to(device)
    dataset = TensorDataset(x, y)
    print(x.shape)
    print(y.shape)
    batch_size = 8
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def resume_training():
    model = Colorization_Model()
    model.load_state_dict(torch.load(args.weights))
    train_loader, val_loader = load_dataset()  # Define your dataset loading logic
    print("Training starting")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    
    for epoch in range(args.epochs, args.epochs + args.epochs):
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
        
        print(f"Epoch [{epoch+1}/{args.epochs + args.epochs}] - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

    new_checkpoint_path = f'./resume_colorization_weights_{args.epochs + args.epochs}.pth'
    torch.save(model.state_dict(), new_checkpoint_path)
    print("Training completed!")

if __name__ == "__main__":
    if args.resume:
        if args.weights == '':
            resume_training()
        else:
            print(f"please enter valid weights path to resume training")
    else:
        run()
