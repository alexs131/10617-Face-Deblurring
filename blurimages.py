import cv2
import os
from PIL import Image

def pixelMosaicing():
    return 0


def pixelate_images_from_folder(folder):
    for filename in os.listdir(folder):
        img = Image.open(folder+filename)
        # Resize smoothly down to 32x32 pixels
        imgSmall = img.resize((16, 16), resample=Image.BILINEAR)

        # Scale back up using NEAREST to original size
        result = imgSmall.resize(img.size, Image.NEAREST)

        # Save
        result.save('lfwcrop_color/faces_pixelated/pixelated' + filename)


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        #cv2.namedWindow(winname = "original", flags = cv2.WINDOW_NORMAL)
        #cv2.imshow(winname = "original",mat=img)
        #cv2.waitKey(0)
        if img is not None:
            blurred = cv2.GaussianBlur(src = img, ksize = (7, 7), sigmaX = 12)
            cv2.imwrite(os.path.join('lfwcrop_color/faces_blurred', 'blurred'+filename), blurred)
            images.append(img)
           #cv2.namedWindow(winname = "blurred", flags = cv2.WINDOW_NORMAL)
            #cv2.imshow(winname = "blurred", mat = blurred)
    #cv2.waitKey(0)

    return images

if __name__ == '__main__':
    print(len(load_images_from_folder('lfwcrop_color/faces')))
    #pixelate_images_from_folder('lfwcrop_color/faces/')
