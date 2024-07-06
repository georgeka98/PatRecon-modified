import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import torchio as tio 

class TeethDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        cbct_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 1])

        # Load the image file (png)
        image = Image.open(img_name).convert('RGB')

        # Load the corresponding bin file
        cbct = self.load_bin_file(cbct_name)

        if self.transform:
            image = self.transform(image)
            cbct = self.transform(cbct)

        sample = {'image': image, 'cbct': cbct}

        return sample

    def load_bin_file(self, file_path):
        with open(file_path, 'rb') as f:
            data = f.read()

        # Open nii.gz file to get the dimensions
        cbct = tio.ScalarImage(file_path.replace('.bin', '.nii.gz'))
        cbct_shape = np.array(cbct)[0].shape
        print(f"CBCT shape: {cbct_shape}")

        # Assuming the bin file contains raw pixel data
        # Adjust dtype and reshape as per your data format
        image_array = np.frombuffer(data, dtype=np.float32)
        image_array = image_array.reshape(cbct_shape)

        # Normalize the data to the range [0, 255] if needed
        image_array = ((image_array - image_array.min()) / (image_array.max() - image_array.min()) * 255).astype(np.uint8)

        # Convert to a PIL Image
        # Handle 3D data correctly by selecting a single slice or creating a montage
        if image_array.ndim == 3:
            image = Image.fromarray(image_array[:, :, 0])  # Using the first slice for now
        else:
            image = Image.fromarray(image_array)

        return image
