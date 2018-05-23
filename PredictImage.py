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
        print("IMAGE_PATHS!\n" + str(image_paths) + "\n")
        warp_img, orig_img = add_images_to_numpy_array(image_paths)
        predicted_images = self.evaluate_model.evaluate_B(orig_img)

        for i, pred_img in enumerate(predicted_images):
            np.save(save_path + "/" + str(model) + "_" + str(i), pred_img)



# Should be called like:
# python3 PredictedImage.py  [directory for input images]  [directory path to save arrays]  [model epochs]
if __name__=='__main__':
    model = sys.argv[3]
    predicted_images = PredictedImage(model)

    img_dir = sys.argv[1]
    save_dir = sys.argv[2]
    filenames = []
    for(dirpath, dirnames, files) in walk(img_dir):
        for filename in files:
            filenames.append(dirpath + filename)
    predicted_images.save_image(filenames, save_dir, model)