import pickle
import numpy as np
import matplotlib.pyplot as plt

class TrainingEvaluation:
    def __init__(self):
        self.historyA = self.readHistory(model='A')
        self.historyB = self.readHistory(model='B')

    def readHistory(self, model='A'):
        filename = "./history/history/trainHistoryDict_AE_specialized_noPreTrain_" + model + "_10000ep" #"_v2" # _specialized _noPreTrain
        history = pickle.load(open(filename, "rb"))
        return history

    '''
    def plot_model_history(self):
        #fig, axs = plt.subplots(1,2,figsize=(15,5))
        fig, axs = plt.subplots(1, 2)
        # summarize history for loss
        axs[1].plot(range(1,len(self.historyA['loss'])+1),self.historyA['loss'])
        #axs[1].plot(range(1,len(self.historyA['val_loss'])+1),self.historyA['val_loss'])
        axs[1].set_title('Model Loss')
        axs[1].set_ylabel('Loss')
        axs[1].set_xlabel('Epoch')
        axs[1].set_xticks(np.arange(1,len(self.historyA['loss'])+1),len(self.historyA['loss'])/10)
        axs[1].legend(['train', 'val'], loc='best')
        plt.show()
        #plt.suptitle(name)
        #plt.savefig("images/modelHistory_"+ name + ".pdf", dpi=300, format="pdf", bbox_inches="tight")

        # The data is in forms of row vectors
        '''

    def plot_model_history_A(self):
        #plt.plot(self.historyB['loss'][0])
        count = 0
        '''
        val_loss = list()
        for item in self.historyA['val_loss']:
            count += 1
            val_loss.append(item[0])
            val_loss.append(item[0])
        plt.plot(val_loss)
        '''
        loss = list()
        min = 99
        bestC = 0

        for i in range(2000):
            loss.append(0.01)

        for item in self.historyA['loss']:
            count += 2
            loss.append(item[1])
            loss.append(item[1])
            if item[1] < min:
                bestC = count
                min = item[1]
            print('Item: ' + str(item[0]) )
        print('Shape: ' + str(self.historyB['loss'][0]) + ' count: ' + str(count))
        plt.plot(loss)
        #plt.plot(self.historyA['val_loss'])

        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('epoch')
        #plt.legend(['train', 'test'], loc='upper left')
        plt.legend(['Val_loss','train'], loc='upper left')
        y1 = loss[3500]
        y2 = loss[9999]
        dy = y2 - y1
        k = dy / 500
        print('K: ' + str(min) + ' Best C: ' + str(bestC+2000) + ' last C: ' + str(y2))
        plt.show()



if __name__=='__main__':
    run = TrainingEvaluation()
    run.plot_model_history_A()