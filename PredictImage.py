from EvaluateModel import EvaluateModel
import numpy as np
import sys
from os import walk
import cv2

class PredictedImage():
    def __init__(self, model_name = False):
        if model_name:
            self.evaluate_model = EvaluateModel(model_name)
        else:
            self.evaluate_model = EvaluateModel("")

    def save_image(self, img_path, save_path):
        print(img_path)
        img = cv2.imread(img_path)
        print(img)
        img  = img / 255.0
        np.save(save_path, self.evaluate_model.evaluate_B(img))


if __name__=='__main__':
    if len(sys.argv) > 3:
        model = sys.argv[2]
        predict_image = PredictedImage(model)
    else:
        predict_image = PredictedImage()

    img_dir = sys.argv[1]
    save_dir = sys.argv[2]
    filenames = []
    for(dirpath, dirnames, files) in walk(img_dir):
        for filename in files:
            filenames.append(dirpath + "/"+ filename)
    for file_path in filenames:
        predict_image.save_image(file_path, save_dir)