import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from net import ReconNet  # Ensure that this module is correctly implemented and accessible
from data_loader import get_train_val_data_loaders


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the training function
def train(model, train_loader, criterion, optimizer, epoch):
    print(device)
    model.train()
    running_loss = 0.0
    for i, batch in enumerate(train_loader):
        inputs, targets = batch['image'], batch['cbct']
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch [{epoch}], Loss: {running_loss / len(train_loader)}')

# Define the validation function
def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    print(f'Validation Loss: {val_loss / len(val_loader)}')
    return val_loss / len(val_loader)

# Main function to train the model
def main():
    parser = argparse.ArgumentParser(description='Train X-ray to CBCT model')
    
    # Required arguments
    parser.add_argument('--train_file', type=str, required=True, help='Path to the training CSV file')
    parser.add_argument('--val_file', type=str, required=True, help='Path to the validation CSV file')
    parser.add_argument('--data_root', type=str, required=True, help='Root directory containing data')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_views', type=int, default=1, help='Number of views')
    parser.add_argument('--input_size', type=int, default=128, help='Input size')
    parser.add_argument('--output_size', type=int, default=128, help='Output size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')

    args = parser.parse_args()

    # Set random seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # Device configuration
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get data loaders
    print(args.train_file, args.val_file)
    train_loader, val_loader = get_train_val_data_loaders(args.train_file, args.val_file, args)

    # Define model, loss function, and optimizer
    model = ReconNet(in_channels=args.num_views, out_channels=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        train(model, train_loader, criterion, optimizer, epoch)
        val_loss = validate(model, val_loader, criterion)

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print('Model saved.')

if __name__ == '__main__':
    main()

print(os.path.abspath(os.getcwd()))
# Example usage:

# python3 train.py --data_root /data --train_file /train.csv --val_file /val.csv --batch_size 8 
# --epochs 100 --learning_rate 1e-4 --num_views 1 --input_size 128 --output_size 128 
# --num_workers 4

# python3 train.py --data_root /data --train_file /train.csv --val_file /val.csv --batch_size 8 --epochs 100 --learning_rate 1e-4 --num_views 1 --input_size 128 --output_size 128 --num_workers 4