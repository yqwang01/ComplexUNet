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
        image_crop = torch.unsqueeze(image_crop / torch.max(fastmri.complex_abs(image_crop)), 0)
        noise = torch.randn_like(image_crop) * 0.1 + 0
        x = image_crop + noise
        y = image_crop

        return (x, y, name)
        

    def __len__(self):
        return len(self.img_path_list)
