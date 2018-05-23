import cv2
import numpy as np
import os
import re

def LoadAndShowResult(path, caption=''):

    # Might need to be of dtyoe='uint8'
    img_array = np.zeros((64, 64*4, 3), dtype='uint8')
    for (root, dirnames, files) in os.walk(path):
        for file_name in files:
            print('root: ' + str(root))
            print('File name: ' + str(file_name))
            if str(file_name) == '.DS_Store':
                continue
            fileIndex = int(re.search(r'\d+', file_name).group())
            print('File index: ' + str(fileIndex))
            start = ((fileIndex) * 64)
            end = (((fileIndex+1) * 64))
            print('start: ' + str(start) + ' end: ' + str(end))
            array = np.load(os.path.join(root, file_name)) * 255
            array = np.array(array, dtype='uint8')
            print('Size: ' + str(np.shape(array)))
            img_array[:, start:end, :] = array


    #print('finall image array: ' + str(img_array[:][:][0]))
    cv2.imshow(caption, img_array)
    cv2.imwrite('./history/spec_no_pre_train_4999/result_4999.jpg', img_array)
    cv2.waitKey(0)

def makeResultMatrix(path):
    img_w = 64*4
    img_h = 64

    img_matrix = np.zeros((img_h*2, img_w*4, 3))
    for (root, dirnames, files) in os.walk(path):
        for file_name in files:
            if str(file_name) == '.DS_Store':
                continue
            filetype = re.sub(r'[0-9]+', '', str(file_name))
            fileIndex = int((int(re.search(r'\d+', file_name).group())+1) / 250)
            print('filetyp: ' + str(filetype) + ' fileIndex: ' + str(fileIndex))

            if filetype == 'result_no_pre_.jpg':
                start_y = 0
                end_y = 64
            else:
                start_y = 64
                end_y = 128
            start_x = (fileIndex-1) * img_w
            end_x = fileIndex * img_w
            img = cv2.imread(os.path.join(root, file_name))
            print('img: ' + str(img))
            img_matrix[start_y:end_y,start_x:end_x,:] = img

    print('finall image array: ' + str(img_matrix[:][:][0]))
    cv2.imshow('', img_matrix)
    cv2.imwrite('./history/trump_orig_new/result_matrix.jpg', img_matrix)
    cv2.waitKey(0)

def create_large_matrix():
    trump = cv2.imread('./history/result_origin.jpg')
    all = cv2.imread('./history/result_matrix.jpg')
    print('all: ' + str(np.shape(all)))


    imgW = 64*4

    b = 10

    new_img = np.ones((64 * 3, (imgW * 4) + 30, 3)) * 255
    new_img[0:64, 0:imgW, :] = trump
    new_img[0:64, b+imgW:imgW*2+b, :] = trump
    new_img[0:64, b+b+imgW*2:imgW*3+b+b, :] = trump
    new_img[0:64, b+b+b+imgW*3:imgW*4+b+b+b, :] = trump

    img1 = all[:, 0:imgW, :]
    img2 = all[:, imgW:imgW * 2, :]
    img3 = all[:, imgW * 2:imgW * 3, :]
    img4 = all[:, imgW * 3:imgW * 4, :]

    new_img[64:, 0:imgW, :] = img1
    new_img[64:, b+imgW:imgW * 2+b, :] = img2
    new_img[64:, b+b+(imgW * 2):(imgW * 3)+b+b, :] = img3
    new_img[64:, b+b+b+(imgW * 3):(imgW * 4)+b+b+b, :] = img4

    cv2.imshow('', new_img)
    cv2.imwrite('./history/finall_result.jpg', new_img)
    cv2.waitKey(0)



def saveNP():
    array1 = np.random.rand(64,64,3)
    np.save('./history/attay1', array1)
    array2 = np.random.rand(64, 64, 3)
    np.save('./history/attay2', array2)
    array3 = np.random.rand(64, 64, 3)
    np.save('./history/attay3', array3)
    array4 = np.random.rand(64, 64, 3)
    np.save('./history/attay4', array4)

def test():
    array1 = np.load('./history/trump_orig/trump_0.npy')
    array2 = np.load('./history/trump_orig/trump_3.npy')
    print('are equal: ' + str(np.array_equal(array1, array2)))

def loadTrump():
    a = cv2.imread('./history/trump_orig_new/trump.npy')
    img_array = cv2.imread((64, 64 * 4, 3), dtype='uint8')
    for i, f in enumerate(a):
        start = ((i) * 64)
        end = (((i + 1) * 64))
        #print('start: ' + str(start) + ' end: ' + str(end))
        img = a[i]
        img = img * 255
        array = np.array(img, dtype='uint8')
        print('Size: ' + str(np.shape(array)))
        img_array[:, start:end, :] = array

    cv2.imshow('', img_array)
    cv2.imwrite('./history/result_origin.jpg', img_array)
    cv2.waitKey(0)

if __name__=='__main__':
    #run = saveNP()
    run = create_large_matrix()
    #run = LoadAndShowResult('./history/spec_no_pre_train_4999')
    #run = test()
    #run = makeResultMatrix('./history/Results')