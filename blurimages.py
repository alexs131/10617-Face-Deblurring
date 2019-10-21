import cv2
import os

def pixelMosaicing():
    return 0


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        cv2.namedWindow(winname = "original", flags = cv2.WINDOW_NORMAL)
        cv2.imshow(winname = "original",mat=img)
        #cv2.waitKey(0)
        if img is not None:
            blurred = cv2.GaussianBlur(src = img, ksize = (5, 5), sigmaX = 12)
            cv2.namedWindow(winname = "blurred", flags = cv2.WINDOW_NORMAL)
            cv2.imshow(winname = "blurred", mat = blurred)
    cv2.waitKey(0)

    return images

if __name__ == '__main__':
    print(len(load_images_from_folder('lfwcrop_color/faces')))
