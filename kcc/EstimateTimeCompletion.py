from keras.callbacks import Callback
import time
import datetime

class EstimateTimeCompletion(Callback):
    def __init__(self):
        super(EstimateTimeCompletion, self).__init__()        
        
    def on_train_begin(self, logs={}):
        self.epoch_num = 0
        self.on_train_begin_time = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.epoch_num += 1
        # Print estimated time to finish
        average_time_per_epoch = (time.time() - self.on_train_begin_time)/self.epoch_num
        epochs_remaining = (self.params['nb_epoch'] - self.epoch_num)
        time_remaining = average_time_per_epoch * epochs_remaining
        estimated_time_completion = (datetime.datetime.now() + datetime.timedelta(seconds=time_remaining)).isoformat(' ')
        print('-----------------------------------------> Estimated finish time: %s'%(estimated_time_completion))
