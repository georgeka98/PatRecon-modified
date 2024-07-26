import numpy as np
import matplotlib.pyplot as plt

def read_npz_file(file_path):
    """
    Reads an .npz file and returns its contents.
    
    Parameters:
    file_path (str): The path to the .npz file.
    
    Returns:
    dict: A dictionary where the keys are the names of the arrays
          and the values are the arrays themselves.
    """
    try:
        with np.load(file_path) as data:
            npz_content = {file: data[file] for file in data.files}
        return npz_content
    except Exception as e:
        print(f"An error occurred while reading the .npz file: {e}")
        return None

def display_cbct_slices(volume, name):
    """
    Displays the top, side, and front slices of a 3D CBCT volume.
    
    Parameters:
    volume (np.ndarray): A 3D numpy array representing the CBCT volume.
    name (str): The name of the array for display purposes.
    """
    if volume.ndim != 3:
        print(f"The array '{name}' is not a 3D volume, it has {volume.ndim} dimensions.")
        return

    # Display the middle slice in each dimension
    middle_axial = volume.shape[0] // 2  # Top (axial) view
    middle_sagittal = volume.shape[1] // 2  # Side (sagittal) view
    middle_coronal = volume.shape[2] // 2  # Front (coronal) view

    plt.figure(figsize=(15, 5))

    # Axial view (top view)
    plt.subplot(1, 3, 1)
    plt.title(f"Axial (Top) View - {name}")
    plt.imshow(volume[middle_axial, :, :], cmap='gray')
    plt.axis('off')

    # Sagittal view (side view)
    plt.subplot(1, 3, 2)
    plt.title(f"Sagittal (Side) View - {name}")
    plt.imshow(volume[:, middle_sagittal, :], cmap='gray')
    plt.axis('off')

    # Coronal view (front view)
    plt.subplot(1, 3, 3)
    plt.title(f"Coronal (Front) View - {name}")
    plt.imshow(volume[:, :, middle_coronal], cmap='gray')
    plt.axis('off')

    plt.show()

# Example usage:
if __name__ == "__main__":
    file_path = 'exp/result/test_prediction.npz'  # Replace with your .npz file path
    npz_content = read_npz_file(file_path)
    
    if npz_content is not None:
        for name, array in npz_content.items():
            print(f"Array name: {name}")
            print(f"Array shape: {array.shape}")
            print(f"Array dtype: {array.dtype}")
            array = array[0]
            if array.ndim == 3:
                display_cbct_slices(array, name)
            else:
                print(f"The array '{name}' is not a 3D volume, it has {array.ndim} dimensions.")
    else:
        print("Failed to read the .npz file.")
