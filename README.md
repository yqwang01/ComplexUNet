
PyTorch implementation of complex convolutional network for Magnetic Reasonance Imaging (MRI) reconstruction 

Please set all your parameters in "configs.py" and run `nohup python -u main.py >> output.log &` on your server.

You can load the new image like the following code:

```python
import numpy as np
import fastmri
from fastmri.data import transforms as T
from matplotlib import pyplot as plt


img_path = "/home/ali/ext2/yqwang/Coursework/BME548/Project/Dataset/singlecoil_val_npz/file1000071/file1000071_15.npz"
img_file = np.load(img_path, allow_pickle=True)
kspace = img_file['img']
img_file.close()
kspace2 = T.to_tensor(kspace)      # Convert from numpy array to pytorch tensor
image = fastmri.ifft2c(kspace2)           # Apply Inverse Fourier Transform to get the complex image
image_abs = fastmri.complex_abs(image)   # Compute absolute value to get a real image
plt.imshow(np.abs(image_abs.numpy()), cmap='gray')
plt.show()
```

If you find this code useful, please cite the following paper:

@article{el2020deep,
  title={Deep complex convolutional network for fast reconstruction of 3D late gadolinium enhancement cardiac MRI},
  author={El-Rewaidy, Hossam and Neisius, Ulf and Mancio, Jennifer and Kucukseymen, Selcuk and Rodriguez, Jennifer and Paskavitz, Amanda and Menze, Bjoern and Nezafat, Reza},
  journal={NMR in Biomedicine},
  volume={33},
  number={7},
  pages={e4312},
  year={2020},
  publisher={Wiley Online Library}
}
