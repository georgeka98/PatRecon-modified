import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

class MedReconDataset(Dataset):
    """ 3D Reconstruction Dataset."""
    def __init__(self, file_list, data_root, num_views, input_size, output_size, transform=None):
        self.df = pd.read_csv(file_list)
        self.data_root = data_root
        self.transform = transform
        self.num_views = num_views
        self.input_size = input_size
        self.output_size = output_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        projs = np.zeros((self.input_size, self.input_size, self.num_views), dtype=np.uint8)

        # Load the panoramic X-ray image
        proj_path = self.df.iloc[idx]['view_0']  # Assuming 'view_0' is the column name
        proj_path = os.path.join(self.data_root, proj_path)
        proj = Image.open(proj_path).resize((self.input_size, self.input_size))
        projs[:, :, 0] = np.array(proj)

        if self.transform:
            projs = self.transform(projs)

        # Load the 3D CBCT image
        image_path = self.df.iloc[idx]['3d_model']
        image_path = os.path.join(self.data_root, image_path)
        image = np.fromfile(image_path, dtype=np.float32)
        image = np.reshape(image, (-1, self.output_size, self.output_size))
        image = image - np.min(image)
        image = image / np.max(image)
        assert((np.max(image) - 1.0 < 1e-3) and (np.min(image) < 1e-3))

        image = torch.from_numpy(image)
        return (projs, image)

