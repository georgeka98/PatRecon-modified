import os
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms

# Assuming TeethDataset is defined in the same file or imported correctly
from dataset import TeethDataset  # Import the TeethDataset class

def get_train_val_data_loaders(train_file, val_file, args):
    # Ensure paths are correct
    train_file_path = os.path.join(args.data_root, train_file)
    val_file_path = os.path.join(args.data_root, val_file)
    
    # Define any transformations if needed
    transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor()
    ])

    # Create datasets
    train_dataset = TeethDataset(csv_file=train_file_path, root_dir=args.data_root, transform=transform)
    val_dataset = TeethDataset(csv_file=val_file_path, root_dir=args.data_root, transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_loader, val_loader





def get_data_loader(file_list, data_root, num_views, 
					input_size, output_size, transform,
					batch_size, train, num_workers):

	dataset = MedReconDataset(file_list=file_list,
				        data_root=data_root,
				        num_views=num_views,
				        input_size=input_size,
				        output_size=output_size,
				        transform=transform)

	loader = DataLoader(dataset=dataset, 
				        batch_size=batch_size, 
				        shuffle=train,
				        num_workers=num_workers, 
				        pin_memory=True)

	return loader