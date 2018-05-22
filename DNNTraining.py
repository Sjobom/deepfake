import DNN
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
import pickle
import numpy as np
import gc
from tensorflow.python.keras.models import load_model
from sklearn.utils import shuffle


class Train_DNN:
    def __init__(self, DIM_ENCODER=1024, lr=5e-5):
        self.model = DNN.DeepModel(DIM_ENCODER, lr)
        self.DIM_ENCODER = DIM_ENCODER
        self.lr = lr

        self.autoencoder_A = self.model.autoencoder_1
        self.autoencoder_B = self.model.autoencoder_2

    # For training the data on general faces in order for the network to be able to encode/ decode the
    # general patterns of a human face, this might be good since the small amount of actual data we have
    def preTraining(self, inputX, targetsY, epochs=5):

        #inputX, targetsY = shuffle(inputX, targetsY, random_state=0)

        lenValDat = int(round(np.shape(inputX)[0]/10))
        valX = inputX[0:lenValDat]
        valY = targetsY[0:lenValDat]
        triningX = inputX[lenValDat:-1]
        trainingY = targetsY[lenValDat:-1]

        best_val_loss_A = 99
        best_val_loss_B = 99
        val_rounds_A = 0;
        val_rounds_B = 0;

        earlyStopping = 4

        # Train the two networks
        history_A = dict()
        history_B = dict()
        history_A['loss'] = []; history_A['val_loss'] = []
        history_B['loss'] = []; history_B['val_loss'] = []
        for epoch in range(epochs):
            epoch_history_A = self.autoencoder_A.fit(inputX, targetsY, epochs=2, batch_size=100, validation_data=(valX, valY))
            epoch_history_B = self.autoencoder_B.fit(inputX, targetsY, epochs=2, batch_size=100, validation_data=(valX, valY))

            # Store history
            history_A['loss'].append(epoch_history_A.history['loss'])
            history_B['loss'].append(epoch_history_B.history['loss'])
            history_A['val_loss'].append(epoch_history_A.history['val_loss'])
            history_B['val_loss'].append(epoch_history_B.history['val_loss'])

            val_loss_A = epoch_history_A.history['val_loss'][-1]
            val_loss_B = epoch_history_B.history['val_loss'][-1]
            # Write val_loss and loss to file
            if val_loss_A < best_val_loss_A:
                best_val_loss_A = val_loss_A
                val_rounds_A = 0
                # Save best A
                self.autoencoder_A.save('./model/autoencoderA_preTrain.hdf5')
                # Save history_A to file
                with open("./history/trainHistoryDict_AE_A", "wb") as file_pi:
                    pickle.dump(history_A, file_pi)
            else:
                val_rounds_A += 1
            if val_loss_B < best_val_loss_B:
                best_val_loss_B = val_loss_B
                val_rounds_B = 0
                # Save best B
                self.autoencoder_B.save('./model/autoencoderB_preTrain.hdf5')
                # Save history_B to file
                with open("./history/trainHistoryDict_AE_B", "wb") as file_pi:
                    pickle.dump(history_B, file_pi)
            else:
                val_rounds_B += 1
            # Test if early stopping should happen
            if val_rounds_A > earlyStopping and val_rounds_B > earlyStopping:
                break


        # Save the trained networks to file
        #self.autoencoder_A.save('autoencoderA_preTrain.hdf5')
        #self.autoencoder_B.save('autoencoderB_preTrain.hdf5')

        # Cleanup
        self.model.delModel()
        del self.model
        del self.autoencoder_A
        del self.autoencoder_B
        gc.collect()

    def train_on_A_and_B(self, inputX_A, targetsY_A, inputX_B, targetsY_B, epochs):

        del self.autoencoder_A
        del self.autoencoder_B
        gc.collect()

        self.autoencoder_A = load_model('./model/autoencoderA_preTrain.hdf5')
        self.autoencoder_B = load_model('./model/autoencoderB_preTrain.hdf5')

        # Train the two networks
        history_A = dict()
        history_B = dict()
        history_A['loss'] = []
        history_B['loss'] = []
        for epoch in range(epochs):
            epoch_history_A = self.autoencoder_A.fit(inputX_A, targetsY_A, epochs=2, batch_size=20)
            epoch_history_B = self.autoencoder_B.fit(inputX_B, targetsY_B, epochs=2, batch_size=20)

            # Store history
            history_A['loss'].append(epoch_history_A.history['loss'])
            history_B['loss'].append(epoch_history_B.history['loss'])

            # Write val_loss and loss to file
            loss_A = epoch_history_A.history['loss'][-1]
            loss_B = epoch_history_B.history['loss'][-1]
            save = [249, 499, 749, 999]
            print("Epoch " + str(epoch))
            if epoch in save:
                self.autoencoder_A.save('./model/autoencoderA_specialized_e' + str(epoch) + '.hdf5')
                # Save history_A to file
                with open("./history/trainHistoryDict_AE_specialized_A", "wb") as file_pi:
                    pickle.dump(history_A, file_pi)
            if epoch in save:
                self.autoencoder_B.save('./model/autoencoderB_specialized_e' + str(epoch) + '.hdf5')
                # Save history_B to file
                with open("./history/trainHistoryDict_AE_specialized_B", "wb") as file_pi:
                    pickle.dump(history_B, file_pi)


        # Cleanup
        del self.autoencoder_A
        del self.autoencoder_B
        del history_A
        del history_B
        gc.collect()
    '''
    def train_on_A_and_B_old(self, inputX_A, targetsY_A, inputX_B, targetsY_B):



        callbacks_list1, callbacks_list2 = self.defineCallBacks()
        # train on the spesific data
        history_A = self.autoencoder_A.fit(inputX_A, targetsY_A, epochs=10000, batch_size=100, callbacks=callbacks_list1, validation_split=0.1)
        history_B = self.autoencoder_B.fit(inputX_B, targetsY_B, epochs=10000, batch_size=100, callbacks=callbacks_list2, validation_split=0.1)

        self.autoencoder_A.save('autoencoderA_preTrain.hdf5')
        self.autoencoder_B.save('autoencoderB_preTrain.hdf5')

        # Save history_A to file
        with open("./history/trainHistoryDict_AE_A", "wb") as file_pi:
            pickle.dump(history_A.history, file_pi)

        # Save history_B to file
        with open("./history/trainHistoryDict_AE_B", "wb") as file_pi:
            pickle.dump(history_B.history, file_pi)

        # Cleanup
        del self.autoencoder_A
        del self.autoencoder_B
        del history_A
        del history_B
        gc.collect()
    '''

    def defineCallBacks(self, earlyStopping=True, ):
        # Define csv logger, for logging loss and acc for every epoch
        csv_logger1 = CSVLogger("./logs/training_autoencoder_A dim-" + str(self.DIM_ENCODER) + " lr-" + str(self.lr) + ".log", separator=',', append=True)
        csv_logger2 = CSVLogger("./logs/training_autoencoder_B dim-" + str(self.DIM_ENCODER) + " lr-" + str(self.lr) + ".log", separator=',', append=True)

        # define the checkpoint, for saving models
        filepath1 = "./model/model_1_-{epoch:02d}-{val_loss:.2f}-{loss:.2f}.hdf5"
        filepath2 = "./model/model_2_-{epoch:02d}-{val_loss:.2f}-{loss:.2f}.hdf5"
        #filepath2 = "model_2_-{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}-{acc:.2f}-{loss:.2f}.hdf5"
        checkpoint1 = ModelCheckpoint(filepath1, monitor='loss', verbose=1, save_best_only=True, mode='min',
                                      period=2)
        checkpoint2 = ModelCheckpoint(filepath2, monitor='loss', verbose=1, save_best_only=True, mode='min',
                                      period=2)

        # Early stopping
        earlystop = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=5, verbose=0, mode='auto')
        callbacks_list1 = [earlystop, checkpoint1, csv_logger1]
        callbacks_list2 = [earlystop, checkpoint2, csv_logger2]

        callbacks_list1 = [earlystop, csv_logger1]
        callbacks_list2 = [earlystop, csv_logger2]

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
    train_X_A = np.random.rand(20, 64, 64, 3)
    train_Y_A = np.random.rand(20, 64, 64, 3)
    train_X_B = np.random.rand(20, 64, 64, 3)
    train_Y_B = np.random.rand(20, 64, 64, 3)
    run.train_on_A_and_B(train_X_A, train_Y_A, train_X_B, train_Y_B, 5)

    #run.preTraining(train_X, train_Y)



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
