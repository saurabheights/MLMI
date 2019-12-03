
from typing import List

from callbacks.callbacks import Callbacks, CallbackMode
from dataset.BaseDataset import BaseDataset


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


def generate_callbacks(arguments: dict,
                       dataset: BaseDataset,
                       device,
                       outdir) -> List[Callbacks]:
    callbacks = []
    """
    Create and add all callbacks instances to `callbacks` list 
    """

    return callbacks