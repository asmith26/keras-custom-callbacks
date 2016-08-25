from keras.callbacks import Callback
import matplotlib.pyplot as plt

class PlotLossAcc(Callback):
    def __init__(self, plot_dir, plot_interval=1):
        self.plot_dir = plot_dir
        self.plot_interval = plot_interval
        super(PlotLossAcc, self).__init__()        
        
    def on_train_begin(self, logs={}):
        self.val_losses = []
        self.val_acc = []
        self.epoch_num = 0

    def on_epoch_end(self, epoch, logs={}):
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))
        self.epoch_num += 1
        
        if self.epoch_num % self.plot_interval == 0 :
            plt.figure()
            plt.subplot(121)
            plt.plot(self.val_losses)
            plt.title('Val_Loss Vs Epoch')
            plt.xlabel('Epoch'); plt.ylabel('Val_Loss')
            plt.subplot(122)
            plt.plot(self.val_acc)
            plt.title('Acc Vs Epoch')
            plt.xlabel('Epoch'); plt.ylabel('Val_Acc')
            plt.ylim(0,1)
            plt.savefig('{}/val-loss_val-acc.png'.format(self.plot_dir))
            plt.close()
