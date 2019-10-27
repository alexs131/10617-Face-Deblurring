import numpy as np
import math
import numpy
import sys

# from scipy import signal
# from scipy import ndimage
#
# import gauss

def psnr(hr_image, sr_image, hr_edge=0):
    # assume RGB image
    hr_image_data = np.array(hr_image)
    if hr_edge > 0:
        hr_image_data = hr_image_data[hr_edge:-hr_edge, hr_edge:-hr_edge].astype('float32')

    sr_image_data = np.array(sr_image).astype('float32')

    diff = sr_image_data - hr_image_data
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff ** 2.))
    return 20 * math.log10(255.0 / rmse)


# def ssim(img1, img2, cs_map=False):
#     """Return the Structural Similarity Map corresponding to input images img1
#     and img2 (images are assumed to be uint8)
#
#     This function attempts to mimic precisely the functionality of ssim.m a
#     MATLAB provided by the author's of SSIM
#     https://ece.uwaterloo.ca/~z70wang/research/ssim/ssim_index.m
#     """
#     img1 = img1.astype(numpy.float64)
#     img2 = img2.astype(numpy.float64)
#     size = 11
#     sigma = 1.5
#     window = gauss.fspecial_gauss(size, sigma)
#     K1 = 0.01
#     K2 = 0.03
#     L = 255  # bitdepth of image
#     C1 = (K1 * L) ** 2
#     C2 = (K2 * L) ** 2
#     mu1 = signal.fftconvolve(window, img1, mode='valid')
#     mu2 = signal.fftconvolve(window, img2, mode='valid')
#     mu1_sq = mu1 * mu1
#     mu2_sq = mu2 * mu2
#     mu1_mu2 = mu1 * mu2
#     sigma1_sq = signal.fftconvolve(window, img1 * img1, mode='valid') - mu1_sq
#     sigma2_sq = signal.fftconvolve(window, img2 * img2, mode='valid') - mu2_sq
#     sigma12 = signal.fftconvolve(window, img1 * img2, mode='valid') - mu1_mu2
#     if cs_map:
#         return (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
#                                                              (sigma1_sq + sigma2_sq + C2)),
#                 (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
#     else:
#         return ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
#                                                             (sigma1_sq + sigma2_sq + C2))