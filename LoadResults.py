import cv2
import numpy as np

def LoadAndShowResult(path, caption=''):
    img = np.load(path)
    cv2.imshow(caption, img)
    cv2.waitkey(0)
