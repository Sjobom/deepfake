from EvaluateModel import EvaluateModel
import numpy as np
import sys
from os import walk
from lib.training_data import add_images_to_numpy_array
import cv2

class PredictedImage():
    def __init__(self, model_name = False):
        if model_name:
            self.evaluate_model = EvaluateModel(model_name)
        else:
            self.evaluate_model = EvaluateModel("")

    def save_image(self, image_paths, save_path, model):
        #img = cv2.imread(img_path)
        #img  = img / 255.0

        warp_img, orig_img = add_images_to_numpy_array(image_paths)
        print (orig_img)
        predicted_image = self.evaluate_model.evaluate_B(orig_img)
        print(predict_image)
        np.save(save_path, predicted_image)


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
    predict_image.save_image(filenames, save_dir)