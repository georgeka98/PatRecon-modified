import os
import numpy as np
import nibabel as nib
from PIL import Image

def load_images_from_folder(folder):
    images = []
    for filename in sorted(os.listdir(folder)):
        img_path = os.path.join(folder, filename)
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        img_array = np.array(img)
        images.append(img_array)
    return np.stack(images, axis=0)  # Stack into a 3D array

def create_nifti(image_volume, output_file, voxel_size=(1.0, 1.0, 1.0)):
    # Create an affine transformation matrix
    affine = np.eye(4)
    affine[0, 0] = voxel_size[0]
    affine[1, 1] = voxel_size[1]
    affine[2, 2] = voxel_size[2]

    # Create a NIfTI image
    nifti_img = nib.Nifti1Image(image_volume, affine)

    # Save the NIfTI image
    nib.save(nifti_img, output_file)

# Example usage
input_folder = "exp/result/sample_1/plane-2"
output_file = "plane-0/plane-2.nii.gz"
voxel_size = (1.0, 1.0, 1.0)  # Adjust as necessary

# Load images and convert to NIfTI file
image_volume = load_images_from_folder(input_folder)
create_nifti(image_volume, output_file, voxel_size)
