from keras.callbacks import Callback
import matplotlib.pyplot as plt

class PlotLossAcc(Callback):
    def __init__(self, plot_dir, plot_interval=1):
        self.plot_dir = plot_dir
        self.plot_interval = plot_interval
        super(PlotLossAcc, self).__init__()        
        
    def on_train_begin(self, logs={}):
        self.losses = []
        self.acc = []
        self.epoch_num = 0
##        self.plot_layers = []
##        for layer in range(len(self.model.layers)):
##            if type(self.model.layers[layer]) == Convolution2D or type(self.model.layers[layer]) == Dense:
##                self.plot_layers.append(layer_index)

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('val_acc'))
        self.epoch_num += 1
        
        if self.epoch_num % self.plot_interval == 0 :
            plt.figure()
            plt.subplot(121)
            plt.plot(self.losses)
            plt.title('Loss Vs Epoch')
            plt.xlabel('Epoch'); plt.ylabel('Loss')
            plt.subplot(122)
            plt.plot(self.acc)
            plt.title('Acc Vs Epoch')
            plt.xlabel('Epoch'); plt.ylabel('Acc')
            plt.ylim(0,1)
            plt.savefig('{}/loss_acc.png'.format(self.plot_dir))
            plt.close()
