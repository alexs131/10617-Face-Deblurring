import numpy as np
import math
import sys
import cv2
from skimage.measure import compare_ssim
from scipy import signal
from scipy import ndimage

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

def ssim(image1,image2):
    grayA = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)
    (score,diff) = compare_ssim(grayA,grayB,full=True)
    diff = (diff*255).astype("uint8")
    return score
def ssim1(image1,image2):
    image1 = np.asarray(image1)
    image2 = np.asarray(image2)
    (score,diff) = compare_ssim(image1,image2,full=True,multichannel=True)
    return score
