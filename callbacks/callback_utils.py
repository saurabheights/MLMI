
from typing import List

from callbacks.callbacks import Callbacks, CallbackMode


def run_callbacks(callbacks: List[Callbacks],
                  model,
                  mode: CallbackMode,
                  epoch=None):
    for callback in callbacks:
        callback.set_model(model)
        if mode == CallbackMode.ON_EPOCH_BEGIN:
            callback.on_epoch_begin(epoch)
        elif mode == CallbackMode.ON_EPOCH_END:
            callback.on_epoch_end(epoch)
        elif mode == CallbackMode.ON_TRAIN_BEGIN:
            callback.on_train_begin()
        elif mode == CallbackMode.ON_TRAIN_END:
            callback.on_train_end()
        else:
            raise ValueError('Unknown mode value')