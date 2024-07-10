import os
import nibabel as nib
import numpy as np

def convert_nii_to_bin(nii_path, bin_path):
    # Load the NIfTI file
    nii_img = nib.load(nii_path)
    data = nii_img.get_fdata()

    # Ensure data is in the correct format (e.g., float32)
    data = data.astype(np.float32)

    # Save as binary
    data.tofile(bin_path)

# Directory containing the NIfTI files
nii_dir = '/Volumes/Georges NVME 2 1/cbct'
# Directory to save the binary files
bin_dir = '/Volumes/Georges NVME 2 1/cbct_bin'

if not os.path.exists(bin_dir):
    os.makedirs(bin_dir)

# delete both val and train csv files
if os.path.exists(bin_dir+'/../train.csv'):
    os.remove(bin_dir+'/../train.csv')
if os.path.exists(bin_dir+'/../val.csv'):
    os.remove(bin_dir+'/../val.csv')

# training
limit = 50
# write the title of the bin file corresponding to the png to the csv file
with open(bin_dir+'/../train.csv', 'a') as f:
    f.write("view_0,3d_model" + '\n')
for filename in os.listdir(nii_dir):
    print('writing to csv')
    if limit == 0:
        break
    if filename.endswith('.nii.gz'):
        bin_filename = filename.replace('.nii.gz', '.bin')
        png_filename = filename.replace('.nii.gz', '.png')
        with open(bin_dir+'/../train.csv', 'a') as f:
            f.write(png_filename + ',' + bin_filename + '\n')
        limit -= 1

# # validation
limit = 10
offset = 51
# write the title of the bin file corresponding to the png to the csv file
with open(bin_dir+'/../val.csv', 'a') as f:
    f.write("view_0,3d_model" + '\n')
for filename in os.listdir(nii_dir):
    if offset == 0:
        if limit == 0:
            break
        if filename.endswith('.nii.gz'):
            bin_filename = filename.replace('.nii.gz', '.bin')
            png_filename = filename.replace('.nii.gz', '.png')
            with open(bin_dir+'/../val.csv', 'a') as f:
                f.write(png_filename + ',' + bin_filename + '\n')
            limit -= 1
    else:
        offset -= 1

#

# Convert all NIfTI files in the directory
# check if there are 50 bin files in the directory
# if there are, then do not convert the nii files to bin files
if len([name for name in os.listdir(bin_dir) if name.endswith('.bin')]) == 50:
    print('All nii files have been converted to bin files')
    exit()


limit = 100
print('Converting NIfTI files to binary...')
for filename in os.listdir(nii_dir):
    if limit == 0:
        break
    # print(filename, limit, nii_dir)
    if filename.endswith('.nii.gz'):
        nii_path = os.path.join(nii_dir, filename)
        bin_filename = filename.replace('.nii.gz', '.bin')
        bin_path = os.path.join(bin_dir, bin_filename)
        convert_nii_to_bin(nii_path, bin_path)
        print(f'Converted {nii_path} to {bin_path}')
        limit -= 1
