from typing import Tuple, List
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import interpolate
import torch
from torchvision import transforms
import fastmri
from fastmri.data import transforms as T
from configs import config
import os
import logging
import scipy
from scipy.ndimage import rotate, gaussian_filter
from torchvision.transforms.functional import to_tensor

def center_crop(data: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """
    Apply a center crop to the input real image or batch of real images.

    Args:
        data: The input tensor to be center cropped. It should
            have at least 3 dimensions and the cropping is applied along the
            last three dimensions.
        shape: The output shape. The shape should be smaller
            than the corresponding dimensions of data.

    Returns:
        The center cropped image.
    """
    if not (0 < shape[0] <= data.shape[-3] and 0 < shape[1] <= data.shape[-2]):
        raise ValueError("Invalid shapes.")

    w_from = (data.shape[-3] - shape[0]) // 2
    h_from = (data.shape[-2] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    
    return data[..., w_from:w_to, h_from:h_to, :]
# def center_crop(data: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
#     """
#     Apply a center crop to the input real image or batch of real images.
#     Handles data with at least 2 dimensions.
#     """
#     assert len(data.shape) >= 2, "Data needs at least 2 dimensions for cropping"

#     # Adjust indexing for tensors with less than 3 dimensions
#     dim1, dim2 = data.shape[-2], data.shape[-1]
#     if not (0 < shape[0] <= dim1 and 0 < shape[1] <= dim2):
#         raise ValueError("Invalid shapes for cropping.")

#     w_from = (dim1 - shape[0]) // 2
#     h_from = (dim2 - shape[1]) // 2
#     w_to = w_from + shape[0]
#     h_to = h_from + shape[1]

#     if len(data.shape) == 2:  # For 2D images
#         return data[w_from:w_to, h_from:h_to]
#     elif len(data.shape) == 3:  # For 3D data where first dimension could be channels
#         return data[:, w_from:w_to, h_from:h_to]
#     else:  # For full batches or more complex structures
#         return data[..., w_from:w_to, h_from:h_to, :]


def dataset_split(data_dir, cut_ratio_1=0.8, cut_ratio_2=1):
    
    case_list = sorted(os.listdir(data_dir))
    num_case = len(case_list)
    print("The number of total valid patients:{}".format(num_case))
    np.random.shuffle(case_list)

    cut_point_1 = int(cut_ratio_1 * num_case)
    cut_point_2 = int(cut_ratio_2 * num_case)
    train_cases = case_list[: cut_point_1] + case_list[cut_point_2:]
    val_cases = case_list[cut_point_1: cut_point_2]
    
    train_list, val_list = [], []
    for case in train_cases:
        for file_name in os.listdir(os.path.join(data_dir, case)):
            train_list.append(os.path.join(case, file_name))
    for case in val_cases:
        for file_name in os.listdir(os.path.join(data_dir, case)):
            val_list.append(os.path.join(case, file_name))
    
    return train_list, val_list


def get_dataloaders() -> Tuple[DataLoader, DataLoader]:
    """Get the dataset loaders.

    Returns
    -------
    Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]
        The training and validation data loaders
    """
    train_list, val_list = dataset_split(config.data_dir, config.cut_ratio_1, config.cut_ratio_2)
    train_dataset = DenoisingDataset(config.data_dir, train_list, config.img_size)
    val_dataset = DenoisingDataset(config.data_dir, val_list, config.img_size)

    tr_data_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.data_loaders_num_workers
    )

    vld_data_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.data_loaders_num_workers
    )

    return tr_data_loader, vld_data_loader


class DenoisingDataset(Dataset):

    def __init__(self, data_dir, img_path_list, img_size):
        super().__init__()
        
        self.data_dir = data_dir
        self.img_size = img_size
        self.img_path_list = img_path_list
  
        # self.noise_transforms = transforms.Compose([
        #                         # transforms.ToPILImage(),
        #                         # transforms.RandomHorizontalFlip(),
        #                         # transforms.RandomVerticalFlip(),
        #                         # transforms.RandomRotation(degrees=(-30, 30)),
        #                         transforms.GaussianBlur(kernel_size=5),
        #                         # transforms.RandomResizedCrop(self.img_size, scale=(0.8, 1.0)),
        #                         transforms.Resize(self.img_size), 
        #                         transforms.ToTensor(),
        #                         transforms.Normalize(mean=[0.5, 0.5], std=[0.5, 0.5])
        #                     ])
        # self.normal_transforms = transforms.Compose([
        #                         # transforms.ToPILImage(),
        #                         transforms.Resize(self.img_size), 
        #                         transforms.ToTensor(),      
        #                         transforms.Normalize(mean=[0.5, 0.5], std=[0.5, 0.5])
        #                     ])

    def __getitem__(self, idx):

        name = self.img_path_list[idx]
        img_path = os.path.join(self.data_dir, self.img_path_list[idx])
        # img = Image.open(img_path).convert('RGB') # cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img_file = np.load(img_path, allow_pickle=True)
        kspace = img_file['img']
        img_file.close()

        kspace2 = T.to_tensor(kspace)      # Convert from numpy array to pytorch tensor
        image = fastmri.ifft2c(kspace2)
        # image = interpolate(image, size=self.img_size, mode='bilinear', align_corners=True)
        image_crop = center_crop(image, shape=self.img_size)
        #image_crop = torch.unsqueeze(image_crop / torch.max(fastmri.complex_abs(image_crop)), 0)
        image_crop =image_crop/torch.max(fastmri.complex_abs(image_crop))
        image_crop_np=image_crop.numpy()

        img_list = []
        angle_list = [-3, 0, 3]
        for angle in angle_list:
            rotated_image = scipy.ndimage.rotate(image_crop_np, angle=angle, axes=(1, 0), reshape=False, mode='constant', cval=0.0)
            #print(rotated_image.shape)
            img_list.append(rotated_image)
        out_image = np.mean(np.stack(img_list, axis=0), axis=0)
        out_image = T.to_tensor(out_image)
        #image_abs = fastmri.complex_abs(out_image)
        
        x=torch.unsqueeze(out_image,0)
        y=torch.unsqueeze(image_crop,0)
        #print(x.shape)
        #print(y.shape)
        return(x,y,name)
    
    # def __getitem__(self, idx):
    #     name = self.img_path_list[idx]
    #     img_path = os.path.join(self.data_dir, name)
    #     img_file = np.load(img_path, allow_pickle=True)
    #     kspace = img_file['img']
    #     img_file.close()

    #     kspace_tensor = to_tensor(kspace).to(torch.complex64)
    #     image = fastmri.ifft2c(kspace_tensor)
    #     image_abs = complex_abs(image)  # Getting the absolute value of the complex number

    #     image_cropped = self.center_crop(image_abs, self.img_size)  # Ensure center_crop handles dimensions correctly
    #     image_cropped /= image_cropped.max()  # Normalization

    #     rotated_images = [scipy.ndimage.rotate(image_cropped.numpy(), angle=angle, axes=(1, 0), reshape=False, mode='constant', cval=0.0) for angle in [-3, 0, 3]]
    #     averaged_image = np.mean(rotated_images, axis=0)

    #     # Convert back to tensor
    #     output_image = torch.from_numpy(averaged_image).float().unsqueeze(0)  # Add channel dimension

    #     return output_image, output_image.clone(), name
    # def __getitem__(self, idx):
    #      name = self.img_path_list[idx]
    #      img_path = os.path.join(self.data_dir, name)
    #      img_file = np.load(img_path, allow_pickle=True)
    #      kspace = img_file['img']
    #      img_file.close()
    
    #      # Convert k-space to a tensor, ensure it's formatted for fastMRI's expected input
    #      kspace_tensor = torch.tensor(kspace, dtype=torch.complex64)  # Ensuring complex type
    #      # Split into separate real and imaginary parts for fastMRI processing
    #      kspace_split = torch.stack([kspace_tensor.real, kspace_tensor.imag], dim=-1)
         
    #      # Perform the inverse FFT
    #      image = fastmri.ifft2c(kspace_split)
    #      image_abs = fastmri.complex_abs(image)  # Getting the absolute value to form the real image
    
    #      # Here we should ensure the image is processed according to your model's needs
    #      # For simplicity, let's assume we just need to crop it to the target size
    #      image_cropped = center_crop(image_abs, self.img_size)
         
    #      return image_cropped, image_cropped, name
    # def __getitem__(self, idx):
    #     name = self.img_path_list[idx]
    #     img_path = os.path.join(self.data_dir, name)
    #     img_file = np.load(img_path, allow_pickle=True)
    #     kspace = img_file['img']
    #     img_file.close()
    
    #     # Handle complex data correctly for fastMRI
    #     kspace_tensor = torch.view_as_complex(torch.tensor(np.stack([np.real(kspace), np.imag(kspace)], axis=-1), dtype=torch.float32))
    #     image = fastmri.ifft2c(kspace_tensor)  # Perform inverse FFT
    #     image_abs = fastmri.complex_abs(image)  # Get magnitude
    
    #     image_cropped = center_crop(image_abs, self.img_size)  # Ensure cropping is correctly applied

    #     return image_cropped, image_cropped, name


    
    # def __getitem__(self, idx):
    #     name = self.img_path_list[idx]
    #     img_path = os.path.join(self.data_dir, name)
    #     with np.load(img_path, allow_pickle=True) as img_file:
    #         kspace = img_file['img']

    #     # Create a tensor from the k-space data, ensuring it is split into real and imaginary parts
    #     #kspace_tensor = torch.stack([
    #     #     torch.from_numpy(np.real(kspace)).float(),
    #     #     torch.from_numpy(np.imag(kspace)).float()
    #     # ], dim=-1)  # Ensure the last dimension holds real and imaginary parts

    #     # Apply the inverse FFT using fastMRI
    #     image = fastmri.ifft2c(kspace_tensor)
    #     image_abs = fastmri.complex_abs(image)  # Get the magnitude of the complex image

    #     # Crop the image to the desired size
    #     image_cropped = center_crop(image_abs, self.img_size)

    #     return image_cropped, image_cropped, name


    def __len__(self):
        return len(self.img_path_list)


 
