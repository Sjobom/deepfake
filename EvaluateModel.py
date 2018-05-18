from tensorflow.python.keras.models import load_model
from tensorflow.python.keras import Model
import numpy as np

class EvaluateModel():
    def __init__(self):
        self.autoencoder_A = load_model('./model/autoencoder_A.hdf5')
        self.autoencoder_B = load_model('./model/autoencoder_B.hdf5')

    def evaluate_A(self, input_X):
        image_A = self.autoencoder_A.predict(input_X)

        return image_A

    def evaluate_B(self, input_X):
        image_B = self.autoencoder_B.predict(input_X)

        return image_B

# if __name__=='__main__':
#     run = EvaluateModel()
#     input_X = np.random.rand(20, 64, 64, 3)
#     img = run.evaluate_A(input_X)
#     print('image: ' + str(img[1]))