import time

import numpy as np
import matplotlib.pyplot as plt
import tifffile

import pylibtim.libtim as tim
import pylibtim.utils as tim_utils

# VREMS-data used for test https://git.unistra.fr/cymeyer/VREMS-data
IMG = np.array(tifffile.imread('VREMS-data/lucchi/image.tif'), dtype=np.uint8)
# Smaller for faster test
IMG = IMG[0:32, 0:512, 0:512]

fig, axs = plt.subplots(2, 2)

print(IMG.shape, IMG.dtype, IMG.min(), IMG.max())
axs[0, 0].imshow(IMG[16], cmap='gray')
axs[0, 0].set_title('original image')

I = np.swapaxes(np.swapaxes(np.copy(IMG), 0,-1), 0, 1)[:, :, 16:17]
S = I.shape
connexity = tim_utils.FlatSE('make2DN8')
tim.area_filtering(S[0], S[1], S[2], I, 10000, connexity)
axs[0, 1].imshow(I[:, :, 0], cmap='gray')
axs[0, 1].set_title('2D image - area_filtering 10k')

I = np.copy(IMG)
S = I.shape
connexity = tim_utils.FlatSE('make3DN26')
t0 = time.time()
tim.area_filtering(S[0], S[1], S[2], I, 100000, connexity)
t1 = time.time()
print(t1-t0)
axs[1, 0].imshow(I[16], cmap='gray')
axs[1, 0].set_title('3D image - area_filtering 100k')

plt.show()
