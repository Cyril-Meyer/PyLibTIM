import time

import numpy as np
import matplotlib.pyplot as plt
import tifffile

import pylibtim.libtim as tim
import pylibtim.utils as tim_utils

# VREMS-data used for test https://git.unistra.fr/cymeyer/VREMS-data
# IMG = np.array(tifffile.imread('VREMS-data/lucchi/image.tif'), dtype=np.uint8)
# Smaller for faster test
# IMG = IMG[0:32, 0:512, 0:512]

# IMG = np.array(tifffile.imread('JG1/ROI2/ROI2.tif'), dtype=np.uint8)
# z = 180

# IMG = np.array(tifffile.imread('LW4/Stack600.tif'), dtype=np.uint8)
# IMG = IMG[0:150]
# z = 45

IMG = np.array(tifffile.imread('I3/i3.tif'), dtype=np.uint8)
IMG = IMG[0:20]
z = 10

print(IMG.shape, IMG.dtype, IMG.min(), IMG.max())

TEST_AREA_FILTERING = False
TEST_ATTRIBUTE_IMAGE = True

if TEST_AREA_FILTERING:
    fig, axs = plt.subplots(2, 2)

    print(IMG.shape, IMG.dtype, IMG.min(), IMG.max())
    axs[0, 0].imshow(IMG[z], cmap='gray')
    axs[0, 0].set_title('original image')

    I = np.swapaxes(np.swapaxes(np.copy(IMG), 0,-1), 0, 1)[:, :, z:z+1]
    S = I.shape
    connexity = tim_utils.FlatSE('make2DN8')
    tim.area_filtering(S[0], S[1], S[2], I, 10000, connexity)

    print(I.shape, I.dtype, I.min(), I.max())
    axs[0, 1].imshow(I[:, :, 0], cmap='gray')
    axs[0, 1].set_title('2D image - area_filtering 10k')

    I = np.copy(IMG)
    S = I.shape
    connexity = tim_utils.FlatSE('make3DN26')
    t0 = time.time()
    tim.area_filtering(S[0], S[1], S[2], I, 100000, connexity)
    t1 = time.time()

    print(t1-t0)
    print(I.shape, I.dtype, I.min(), I.max())
    axs[1, 0].imshow(I[z], cmap='gray')
    axs[1, 0].set_title('3D image - area_filtering 100k')

    I = np.copy(IMG)
    S = I.shape
    connexity = tim_utils.FlatSE('make3DN26')
    t0 = time.time()
    tim.area_filtering(S[0], S[1], S[2], I, 1000000, connexity)
    t1 = time.time()

    print(t1-t0)
    print(I.shape, I.dtype, I.min(), I.max())
    axs[1, 1].imshow(I[z], cmap='gray')
    axs[1, 1].set_title('3D image - area_filtering 1000k')

    plt.show()


if TEST_ATTRIBUTE_IMAGE:
    fig, axs = plt.subplots(2, 3)

    I = np.copy(IMG)
    I_ATTR = np.zeros((IMG.shape + (7,)), dtype=np.int32)
    S = I.shape
    connexity = tim_utils.FlatSE('make3DN26')
    t0 = time.time()
    tim.attribute_image(S[0], S[1], S[2], I, I_ATTR, connexity)
    t1 = time.time()

    print(t1-t0)
    print(I_ATTR[:, :, :, 0].shape, I_ATTR[:, :, :, 0].dtype, I_ATTR[:, :, :, 0].min(), I_ATTR[:, :, :, 0].max())
    axs[0, 0].imshow(I_ATTR[z, :, :, 0]/I_ATTR[:,:,:,0].max(), cmap='gray')
    axs[0, 0].set_title('3D image - attribute_image area')
    print(I_ATTR[:, :, :, 1].shape, I_ATTR[:, :, :, 1].dtype, I_ATTR[:, :, :, 1].min(), I_ATTR[:, :, :, 1].max())
    axs[0, 1].imshow(I_ATTR[z, :, :, 1]/I_ATTR[:,:,:,1].max(), cmap='gray')
    axs[0, 1].set_title('3D image - attribute_image contrast')
    print(I_ATTR[:, :, :, 2].shape, I_ATTR[:, :, :, 2].dtype, I_ATTR[:, :, :, 2].min(), I_ATTR[:, :, :, 2].max())
    axs[0, 2].imshow(I_ATTR[z, :, :, 2]/I_ATTR[:,:,:,2].max(), cmap='gray')
    axs[0, 2].set_title('3D image - attribute_image contourLength')
    print(I_ATTR[:, :, :, 3].shape, I_ATTR[:, :, :, 3].dtype, I_ATTR[:, :, :, 3].min(), I_ATTR[:, :, :, 3].max())
    axs[1, 0].imshow(I_ATTR[z, :, :, 3]/I_ATTR[:,:,:,3].max(), cmap='gray')
    axs[1, 0].set_title('3D image - attribute_image subNodes')
    print(I.shape, I.dtype, I.min(), I.max())
    axs[1, 2].imshow(I[z, :, :], cmap='gray')
    axs[1, 2].set_title('3D image - original')

    plt.show()
