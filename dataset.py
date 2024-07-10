import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as transforms
import torchio as tio

class TeethDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        cbct_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 1])

        # Load the image file (png)
        image = Image.open(img_name)

        # Load the corresponding bin file
        cbct = self.load_bin_file(cbct_name)
        cbct = torch.from_numpy(cbct).float()

        if self.transform:
            image = self.transform(image)
            cbct = torch.tensor(cbct)

        return image, cbct

    def load_bin_file(self, file_path):
        with open(file_path, 'rb') as f:
            data = f.read()

        # Open nii.gz file to get the dimensions
        cbct = tio.ScalarImage(file_path.replace('.bin', '.nii.gz'))
        cbct_shape = np.array(cbct)[0].shape

        # Assuming the bin file contains raw pixel data
        image_array = np.frombuffer(data, dtype=np.float32)
        image_array = image_array.reshape((cbct_shape[0], cbct_shape[1], cbct_shape[2]))
        return image_array
