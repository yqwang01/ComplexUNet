import os
import h5py
import numpy as np

input_folder = None
output_folder = None
file_list = os.listdir(input_folder)


for file in file_list:
    h5file = os.path.join(input_folder, file)
    hf = h5py.File(h5file)
    volume_kspace = hf['kspace'][()]
    os.makedirs(os.path.join(output_folder, file[:-3]), exist_ok=True)

    mid = volume_kspace.shape[0] // 2
    slice_list = [mid - 2, mid - 1, mid, mid + 1, mid + 2]
    for slice_num in slice_list:
        img = volume_kspace[slice_num]
        npz_file = os.path.join(output_folder, file[:-3], file[:-3] + '_' + str(slice_num) + '.npz')
        img_array = np.array(img)
        np.savez_compressed(npz_file, img=img_array)