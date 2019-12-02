import cv2
import os
from PIL import Image
from pyblur import *


def pixelate_images_from_folder(folder):
    for filename in os.listdir(folder):
        img = Image.open(folder+filename)
        # Resize smoothly down to 32x32 pixels
        imgSmall = img.resize((32, 32), resample=Image.BILINEAR)

         # Scale back up using NEAREST to original size
        result = imgSmall.resize(img.size, Image.NEAREST)
        result.save('lfwcrop_color/faces_pixelated/'+filename)
    return



def defocus_blur(folder):
     for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        img = Image.open(folder+filename)
        #cv2.namedWindow(winname = "original", flags = cv2.WINDOW_NORMAL)
        #cv2.imshow(winname = "original",mat=img)
        #cv2.waitKey(0)
        if img is not None:
            blurred = DefocusBlur(img, 3)
            blurred.save('lfwcrop_color/faces_defocus_blur/' + filename)
def box_blur(folder):
   for filename in os.listdir(folder):
        img = Image.open(folder+filename)
        #cv2.namedWindow(winname = "original", flags = cv2.WINDOW_NORMAL)
        #cv2.imshow(winname = "original",mat=img)
        #cv2.waitKey(0)
        if img is not None:
            blurred = BoxBlur(img, 3)
            blurred.save('lfwcrop_color/faces_box_blur/' + filename)


def linear_motion(folder):
    for i in [0,90]:
        for filename in os.listdir(folder):
           img = Image.open(folder+filename)
            #cv2.namedWindow(winname = "original", flags = cv2.WINDOW_NORMAL)
            #cv2.imshow(winname = "original",mat=img)
           #cv2.waitKey(0)
            z = filename.split('.')
            if img is not None:
               blurred = LinearMotionBlur(img, 7, i, "full")
               blurred.save('lfwcrop_color/faces_linear_motion'+ str(i) + '/' + filename)

def load_images_from_folder(folder):
    seq = [0.5,1.0,1.5,2.0,2.5,3.0]
    for i in seq:
        for filename in os.listdir(folder):
           img = Image.open(folder+filename)
            #cv2.namedWindow(winname = "original", flags = cv2.WINDOW_NORMAL)
            #cv2.imshow(winname = "original",mat=img)
            #cv2.waitKey(0)
            if img is not None:
                blurred = GaussianBlur(img, i)
                blurred.save('lfwcrop_color/faces_blurred' + str(i) + '/' + filename)
            #cv2.namedWindow(winname = "blurred", flags = cv2.WINDOW_NORMAL)
             #cv2.imshow(winname = "blurred", mat = blurred)
     #cv2.waitKey(0)

 if __name__ == '__main__':
     load_images_from_folder('lfwcrop_color/faces/')
     linear_motion('lfwcrop_color/faces/')
     #defocus_blur('lfwcrop_color/faces/')
     #box_blur('lfwcrop_color/faces/')
     pixelate_images_from_folder('lfwcrop_color/faces/')
