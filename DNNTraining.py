import DNN
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
import pickle
import numpy as np

class Train_DNN:
    def __init__(self):
        self.model = DNN.DeepModel()

        self.autoencoder_A = self.model.autoencoder_1
        self.autoencoder_B = self.model.autoencoder_2

    # For training the data on general faces in order for the network to be able to encode/ decode the
    # general patterns of a human face, this might be good since the small amount of actual data we have
    def preTraining(self, inputX, targetsY):
        callbacks_list1, callbacks_list2 = self.defineCallBacks()
        # Train the two networks
        history_A = self.autoencoder_A.fit(inputX, targetsY, epochs=2, batch_size=5, validation_split=0.1)
        history_B = self.autoencoder_B.fit(inputX, targetsY, epochs=2, batch_size=5, validation_split=0.1)
        # Save the trained networks to file
        self.autoencoder_A.save('autoencoderA_preTrain.hdf5')
        self.autoencoder_B.save('autoencoderB_preTrain.hdf5')

        # Save history_A to file
        with open("./pre_trainHistoryDict_AE_A", "wb") as file_pi:
            pickle.dump(history_A.history, file_pi)

        # Save history_B to file
        with open("./pre_trainHistoryDict_AE_B", "wb") as file_pi:
            pickle.dump(history_B.history, file_pi)

    def train_on_A_and_B(self, inputX_A, targetsY_A, inputX_B, targetsY_B):
        callbacks_list1, callbacks_list2 = self.defineCallBacks()
        # train on the spesific data
        history_A = self.autoencoder_A.fit(inputX_A, targetsY_A, epochs=10000, batch_size=100, callbacks=callbacks_list2, validation_split=0.1)
        history_B = self.autoencoder_B.fit(inputX_B, targetsY_B, epochs=10000, batch_size=100, callbacks=callbacks_list2, validation_split=0.1)

        self.autoencoder_A.save('autoencoderA_preTrain.hdf5')
        self.autoencoder_B.save('autoencoderB_preTrain.hdf5')

        # Save history_A to file
        with open("./trainHistoryDict_AE_A", "wb") as file_pi:
            pickle.dump(history_A.history, file_pi)

        # Save history_B to file
        with open("./trainHistoryDict_AE_B", "wb") as file_pi:
            pickle.dump(history_B.history, file_pi)

    def defineCallBacks(self, earlyStopping=True):
        # Define csv logger, for logging loss and acc for every epoch
        csv_logger1 = CSVLogger("training_autoencoderA" + ".log", separator=',', append=True)
        csv_logger2 = CSVLogger("training_autoencoderB" + ".log", separator=',', append=True)

        # define the checkpoint, for saving models
        filepath1 = "model_1_-{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}-{acc:.2f}-{loss:.2f}.hdf5"
        filepath2 = "model_2_-{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}-{acc:.2f}-{loss:.2f}.hdf5"
        checkpoint1 = ModelCheckpoint(filepath1, monitor='loss', verbose=1, save_best_only=True, mode='min',
                                      period=100)
        checkpoint2 = ModelCheckpoint(filepath2, monitor='loss', verbose=1, save_best_only=True, mode='min',
                                      period=100)

        # Early stopping
        earlystop = EarlyStopping(monitor='val_acc', min_delta=0.00001, patience=50, verbose=0, mode='auto')
        callbacks_list1 = [earlystop, checkpoint1, csv_logger1]
        callbacks_list2 = [earlystop, checkpoint2, csv_logger2]

        return callbacks_list1, callbacks_list2

    def predict(self, input_X, ae_A=True, ae_B=True):
        if ae_B and ae_A:
            image_A = self.autoencoder_A.predict(input_X)
            image_B = self.autoencoder_B.predict(input_X)
            return image_A, image_B
        elif ae_B:
            image_B = self.autoencoder_B.predict(input_X)
            return image_B
        elif ae_A:
            image_A = self.autoencoder_A.predict(input_X)
            return image_A


if __name__ == '__main__':
    run = Train_DNN()
    train_X = np.random.rand(20, 64, 64, 3)
    train_Y = np.random.rand(20, 64, 64, 3)
    run.preTraining(train_X, train_Y)



'''

    # functions to plot the model history during training
    
    def readHistory(self, ftype1, ftype2):
        filename = "trainHistoryDict_" + ftype1 + "_" + ftype2
        history = pickle.load(open(filename, "rb"))
        return history
        
     def plot_model_history(self,history,name):
        #fig, axs = plt.subplots(1,2,figsize=(15,5))
        fig, axs = plt.subplots(1, 2)
        # summarize history for accuracy
        axs[0].plot(range(1,len(history['acc'])+1),history['acc'])
        axs[0].plot(range(1,len(history['val_acc'])+1),history['val_acc'])
        axs[0].set_title('Model Accuracy')
        axs[0].set_ylabel('Accuracy')
        axs[0].set_xlabel('Epoch')
        axs[0].set_xticks(np.arange(1,len(history['acc'])+1),len(history['acc'])/10)
        axs[0].legend(['train', 'val'], loc='best')
        # summarize history for loss
        axs[1].plot(range(1,len(history['loss'])+1),history['loss'])
        axs[1].plot(range(1,len(history['val_loss'])+1),history['val_loss'])
        axs[1].set_title('Model Loss')
        axs[1].set_ylabel('Loss')
        axs[1].set_xlabel('Epoch')
        axs[1].set_xticks(np.arange(1,len(history['loss'])+1),len(history['loss'])/10)
        axs[1].legend(['train', 'val'], loc='best')
        #plt.show()
        plt.suptitle(name)
        plt.savefig("images/modelHistory_"+ name + ".pdf", dpi=300, format="pdf", bbox_inches="tight")

        # The data is in forms of row vectors

'''