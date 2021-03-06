import time

import numpy as np
import matplotlib.pyplot as plt
import tifffile

import pylibtim.libtim as tim

# VREMS-data used for test https://git.unistra.fr/cymeyer/VREMS-data
# IMG = np.array(tifffile.imread('VREMS-data/lucchi/image.tif'), dtype=np.uint8)
# Smaller for faster test
# IMG = IMG[0:32, 0:256, 0:256]
# z = 16

IMG = np.array(tifffile.imread('JG1/ROI2/ROI2.tif'), dtype=np.uint8)
IMG = IMG[0:64]
z = 32

# IMG = np.array(tifffile.imread('LW4/Stack600.tif'), dtype=np.uint8)
# IMG = IMG[0:150]
# z = 45

# IMG = np.array(tifffile.imread('I3/i3.tif'), dtype=np.uint8)
# IMG = IMG[0:20]
# z = 10

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
    connexity = tim.C2DN8
    tim.area_filtering(S[0], S[1], S[2], I, 10000, connexity)

    print(I.shape, I.dtype, I.min(), I.max())
    axs[0, 1].imshow(I[:, :, 0], cmap='gray')
    axs[0, 1].set_title('2D image - area_filtering 10k')

    I = np.copy(IMG)
    S = I.shape
    connexity = tim.C3DN26
    t0 = time.time()
    tim.area_filtering(S[0], S[1], S[2], I, 100000, connexity)
    t1 = time.time()

    print(t1-t0)
    print(I.shape, I.dtype, I.min(), I.max())
    axs[1, 0].imshow(I[z], cmap='gray')
    axs[1, 0].set_title('3D image - area_filtering 100k')

    I = np.copy(IMG)
    S = I.shape
    connexity = tim.C3DN26
    t0 = time.time()
    tim.area_filtering(S[0], S[1], S[2], I, 1000000, connexity)
    t1 = time.time()

    print(t1-t0)
    print(I.shape, I.dtype, I.min(), I.max())
    axs[1, 1].imshow(I[z], cmap='gray')
    axs[1, 1].set_title('3D image - area_filtering 1000k')

    plt.show()


if TEST_ATTRIBUTE_IMAGE:
    fig, axs = plt.subplots(2, 2)

    I = np.copy(IMG)
    S = I.shape

    AttributeID = tim.MSER

    print(IMG.shape, IMG.dtype, IMG.min(), IMG.max())
    axs[0, 0].imshow(IMG[z], cmap='gray')
    axs[0, 0].set_title('original image')

    I_ATTR = np.zeros(IMG.shape, dtype=np.int32)
    t0 = time.time()
    tim.attribute_image(S[0], S[1], S[2], I, I_ATTR, tim.C3DN26, AttributeID, tim.NODE)
    t1 = time.time()
    print(t1-t0)
    print(I_ATTR.shape, I_ATTR.dtype, I_ATTR.min(), I_ATTR.max())
    I_ATTR = I_ATTR / I_ATTR.max()
    axs[0, 1].imshow(I_ATTR[z], cmap='gray')
    axs[0, 1].set_title('attribute image node')

    I_ATTR_MAX = np.zeros(IMG.shape, dtype=np.int32)
    t0 = time.time()
    tim.attribute_image(S[0], S[1], S[2], I, I_ATTR_MAX, tim.C3DN26, AttributeID, tim.MAX_PARENTS)
    t1 = time.time()
    print(t1-t0)
    print(I_ATTR_MAX.shape, I_ATTR_MAX.dtype, I_ATTR_MAX.min(), I_ATTR_MAX.max())
    I_ATTR_MAX = I_ATTR_MAX / I_ATTR_MAX.max()
    axs[1, 0].imshow(I_ATTR_MAX[z], cmap='gray')
    axs[1, 0].set_title('attribute image max')

    I_ATTR_MIN = np.zeros(IMG.shape, dtype=np.int32)
    t0 = time.time()
    tim.attribute_image(S[0], S[1], S[2], I, I_ATTR_MIN, tim.C3DN26, AttributeID, tim.MIN_PARENTS)
    t1 = time.time()
    print(t1-t0)
    print(I_ATTR_MIN.shape, I_ATTR_MIN.dtype, I_ATTR_MIN.min(), I_ATTR_MIN.max())
    I_ATTR_MIN = I_ATTR_MIN / I_ATTR_MIN.max()
    axs[1, 1].imshow(I_ATTR_MIN[z], cmap='gray')
    axs[1, 1].set_title('attribute image min')

    '''
    I_ATTR_DIFF = I_ATTR_MAX - I_ATTR
    print(I_ATTR_DIFF.shape, I_ATTR_DIFF.dtype, I_ATTR_DIFF.min(), I_ATTR_DIFF.max())
    axs[1, 1].imshow(I_ATTR_DIFF[z], cmap='gray')
    axs[1, 1].set_title('attribute image diff')
    '''

    tifffile.imsave('image_pp.tif', np.array(I_ATTR*255).astype(np.uint8))
    tifffile.imsave('image_pp_max.tif', np.array(I_ATTR_MAX*255).astype(np.uint8))
    tifffile.imsave('image_pp_min.tif', np.array(I_ATTR_MIN*255).astype(np.uint8))

    plt.show()
