from enum import Enum, unique, auto


@unique
class CallbackMode(Enum):
    ON_TRAIN_BEGIN = auto()
    ON_EPOCH_BEGIN = auto()
    ON_EPOCH_END = auto()
    ON_TRAIN_END = auto()
    ON_NTH_ITERATION = auto()


class Callbacks:
    def __init__(self):
        pass

    def set_model(self, model):
        self.model = model

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def on_train_begin(self):
        pass

    def on_train_end(self):
        pass

    def on_epoch_begin(self, epoch):
        pass

    def on_epoch_end(self, epoch):
        pass

    def on_nth_iteration(self, iteration):
        """
        Useful in for example gan training.
        :param iteration:
        :return:
        """
        pass