import argparse
import torch
from torch.utils.data import DataLoader
from trainer import Trainer_ReconNet
from dataset import TeethDataset
import torchvision.transforms as transforms

def main():
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('--data_root', type=str, required=True, help='Path to the data root directory')
    parser.add_argument('--train_file', type=str, required=True, help='Path to the training CSV file')
    parser.add_argument('--val_file', type=str, required=True, help='Path to the validation CSV file')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--input_size', type=int, default=128, help='Input size of images')
    parser.add_argument('--output_channel', type=int, default=128, help='Output size of images')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--exp', type=str, default='experiment', help='Experiment name')
    parser.add_argument('--arch', type=str, default='ReconNet', help='Model architecture')
    parser.add_argument('--print_freq', type=int, default=10, help='Print frequency')
    parser.add_argument('--output_path', type=str, default='./output', help='Output path')
    parser.add_argument('--resume', type=str, default='best', help='Resume from checkpoint')
    parser.add_argument('--loss', type=str, default='l1', help='Loss function')
    parser.add_argument('--optim', type=str, default='adam', help='Optimizer')
    parser.add_argument('--num_views', type=int, default=1, help='Number of input views')
    parser.add_argument('--init_gain', type=float, default=0.02, help='Initialization gain')
    parser.add_argument('--init_type', type=str, default='normal', help='Initialization type')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
    parser.add_argument('--lr', type=float, default=0.001, help='Adam learning rate')

    args = parser.parse_args()

    # Create datasets and data loaders
    transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
    ])
    train_dataset = TeethDataset(csv_file=args.train_file, root_dir=args.data_root, transform=transform)
    val_dataset = TeethDataset(csv_file=args.val_file, root_dir=args.data_root, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Initialize the trainer
    trainer = Trainer_ReconNet(args)

    # Load from checkpoint if needed
    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load()

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        train_loss = trainer.train_epoch(train_loader, epoch)
        val_loss = trainer.validate(val_loader)
        print(f"Validation Loss: {val_loss:.5f}")
        trainer.save(val_loss, epoch)

if __name__ == '__main__':
    main()
