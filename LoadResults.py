#import cv2
import numpy as np
import os
import re

def LoadAndShowResult(path, caption=''):

    # Might need to be of dtyoe='uint8'
    img_array = np.zeros((64, 64*4, 3), dtype='uint8')
    for (root, dirnames, files) in os.walk(path):
        for file_name in files:
            print('File name: ' + str(file_name))
            fileIndex = int(re.search(r'\d+', file_name).group())
            print('File index: ' + str(fileIndex))
            start = ((fileIndex-1) * 64)
            end = ((fileIndex * 64))
            print('start: ' + str(start) + ' end: ' + str(end))
            array = np.load(os.path.join(root, file_name)) * 255
            #array = np.array(array, dtype='uint8')
            print('array: ' + str(array))
            print('Size: ' + str(np.shape(array)))
            print('Shape: ' + str(np.shape(img_array[:, 0:64, :])))
            img_array[:, start:end, :] = array

    print('finall image array: ' + str(img_array[:][:][0]))
    #cv2.imshow(caption, img)
    #cv2.waitkey(0)

def saveNP():
    array1 = np.random.rand(64,64,3)
    np.save('./history/attay1', array1)
    array2 = np.random.rand(64, 64, 3)
    np.save('./history/attay2', array2)
    array3 = np.random.rand(64, 64, 3)
    np.save('./history/attay3', array3)
    array4 = np.random.rand(64, 64, 3)
    np.save('./history/attay4', array4)

if __name__=='__main__':
    #run = saveNP()
    run = LoadAndShowResult('./history')