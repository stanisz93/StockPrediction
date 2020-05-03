from keras.callbacks import Callback


class NeptuneMonitor(Callback):
    def __init__(self, neptune_experiment, n_batch):
        super().__init__()
        self.exp = neptune_experiment
        self.n = n_batch
        self.current_epoch = 0

    def on_batch_end(self, batch, logs=None):
        x = (self.current_epoch * self.n) + batch
        self.exp.send_metric(
            channel_name='batch end loss', x=x, y=logs['loss'])

    def on_epoch_end(self, epoch, logs=None):
        self.exp.send_metric('epoch end loss', logs['loss'])
        self.exp.send_metric('val epoch end loss', logs['val_loss'])

        self.current_epoch += 1
