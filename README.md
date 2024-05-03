
PyTorch implementation of complex convolutional network for Magnetic Reasonance Imaging (MRI) reconstruction and denoising.

Please set all your parameters in "configs.py" and run `nohup python -u main.py >> output.log &` on your server.

You can load the new image like the following code:

```python
import numpy as np
import fastmri
from fastmri.data import transforms as T
from matplotlib import pyplot as plt


img_path = "./singlecoil_val_npz/file1000071/file1000071_15.npz"
img_file = np.load(img_path, allow_pickle=True)
kspace = img_file['img']
img_file.close()
kspace2 = T.to_tensor(kspace)      # Convert from numpy array to pytorch tensor
image = fastmri.ifft2c(kspace2)           # Apply Inverse Fourier Transform to get the complex image
image_abs = fastmri.complex_abs(image)   # Compute absolute value to get a real image
plt.imshow(np.abs(image_abs.numpy()), cmap='gray')
plt.show()
```

Our code refers to [this repository](https://github.com/helrewaidy/deep-complex-convolutional-network).

You can download the [fastMRI Dataset](https://fastmri.med.nyu.edu/) and make the compressed dataset by running “make_npz.py”.

