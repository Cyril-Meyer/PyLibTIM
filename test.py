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

TEST_AREA_FILTERING = False
TEST_ATTRIBUTE_IMAGE = True

if TEST_AREA_FILTERING:
    fig, axs = plt.subplots(2, 2)

    print(IMG.shape, IMG.dtype, IMG.min(), IMG.max())
    axs[0, 0].imshow(IMG[16], cmap='gray')
    axs[0, 0].set_title('original image')

    I = np.swapaxes(np.swapaxes(np.copy(IMG), 0,-1), 0, 1)[:, :, 16:17]
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
    axs[1, 0].imshow(I[16], cmap='gray')
    axs[1, 0].set_title('3D image - area_filtering 100k')

    I = np.copy(IMG)
    S = I.shape
    connexity = tim_utils.FlatSE('make3DN26')
    t0 = time.time()
    tim.area_filtering(S[0], S[1], S[2], I, 1000000, connexity)
    t1 = time.time()

    print(t1-t0)
    print(I.shape, I.dtype, I.min(), I.max())
    axs[1, 1].imshow(I[16], cmap='gray')
    axs[1, 1].set_title('3D image - area_filtering 1000k')

    plt.show()


if TEST_ATTRIBUTE_IMAGE:
    fig, axs = plt.subplots(2, 3)

    I = np.copy(IMG)
    I_ATTR = np.zeros((IMG.shape + (6,)), dtype=np.float64)
    S = I.shape
    connexity = tim_utils.FlatSE('make3DN26')
    t0 = time.time()
    tim.attribute_image(S[0], S[1], S[2], I, I_ATTR, connexity)
    t1 = time.time()

    print(t1-t0)
    print(I_ATTR.shape, I_ATTR.dtype, I_ATTR.min(), I_ATTR.max())
    axs[0, 0].imshow(I_ATTR[16, :, :, 0]/I_ATTR.max(), cmap='gray')
    axs[0, 0].set_title('3D image - attribute_image area')
    axs[0, 1].imshow(I_ATTR[16, :, :, 1]/I_ATTR.max(), cmap='gray')
    axs[0, 1].set_title('3D image - attribute_image volume')
    axs[0, 2].imshow(I_ATTR[16, :, :, 2]/I_ATTR.max(), cmap='gray')
    axs[0, 2].set_title('3D image - attribute_image contrast')
    axs[1, 0].imshow(I_ATTR[16, :, :, 3]/I_ATTR.max(), cmap='gray')
    axs[1, 0].set_title('3D image - attribute_image contourLength')
    axs[1, 1].imshow(I_ATTR[16, :, :, 4]/I_ATTR.max(), cmap='gray')
    axs[1, 1].set_title('3D image - attribute_image complexity')
    axs[1, 2].imshow(I_ATTR[16, :, :, 5]/I_ATTR.max(), cmap='gray')
    axs[1, 2].set_title('3D image - attribute_image compacity')

    plt.show()
