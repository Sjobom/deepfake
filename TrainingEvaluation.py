import pickle
import numpy as np
import matplotlib.pyplot as plt

class TrainingEvaluation:
    def __init__(self):
        self.historyA = self.readHistory(model='A')
        self.historyB = self.readHistory(model='B')

    def readHistory(self, model='A'):
        filename = "./history/trainHistoryDict_AE_" + model
        history = pickle.load(open(filename, "rb"))
        return history

    def plot_model_history(self):
        #fig, axs = plt.subplots(1,2,figsize=(15,5))
        fig, axs = plt.subplots(1, 2)
        # summarize history for loss
        axs[1].plot(range(1,len(self.historyA['loss'])+1),self.historyA['loss'])
        axs[1].plot(range(1,len(self.historyA['val_loss'])+1),self.historyA['val_loss'])
        axs[1].set_title('Model Loss')
        axs[1].set_ylabel('Loss')
        axs[1].set_xlabel('Epoch')
        axs[1].set_xticks(np.arange(1,len(self.historyA['loss'])+1),len(self.historyA['loss'])/10)
        axs[1].legend(['train', 'val'], loc='best')
        plt.show()
        #plt.suptitle(name)
        #plt.savefig("images/modelHistory_"+ name + ".pdf", dpi=300, format="pdf", bbox_inches="tight")

        # The data is in forms of row vectors

    def plot_model_history_A(self):
        plt.plot(self.historyA['loss'])
        plt.plot(self.historyA['val_loss'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

if __name__=='__main__':
    run = TrainingEvaluation()
    run.plot_model_history_A()