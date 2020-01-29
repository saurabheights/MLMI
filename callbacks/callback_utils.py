from typing import List

import torch

from callbacks.callbacks import Callbacks, CallbackMode
from dataset.BaseDataset import BaseDataset
from metrics.frechet_metric_callback import FrechetInceptionScoreCallback
from metrics.inception_metric_callback import InceptionScoreCallback
from models.utils import get_model
from visualization.gan_embedding_sampler import GanEmbeddingSampler

from visualization.gan_sampler import GanSampler


def run_callbacks(callbacks: List[Callbacks],
                  model,
                  optimizer,
                  mode: CallbackMode,
                  epoch=None,
                  iteration=None):
    for callback in callbacks:
        callback.set_model(model)
        callback.set_optimizer(optimizer)
        if mode == CallbackMode.ON_EPOCH_BEGIN:
            callback.on_epoch_begin(epoch)
        elif mode == CallbackMode.ON_EPOCH_END:
            callback.on_epoch_end(epoch)
        elif mode == CallbackMode.ON_TRAIN_BEGIN:
            callback.on_train_begin()
        elif mode == CallbackMode.ON_TRAIN_END:
            callback.on_train_end()
        elif mode == CallbackMode.ON_NTH_ITERATION:
            callback.on_nth_iteration(iteration)
        else:
            raise ValueError('Unknown mode value')


def batch_normalize_transform(mean, std):
    def f(batch_sample):
        tmp = batch_sample.mul(0.5).add(0.5)  # Tanh layer of model
        tmp = (batch_sample - mean[0]) / std[0]  # ToDo Make it work for all images.
        return tmp

    return f


def generate_inception_metric_callback(callback_args, device, outdir):
    mode = callback_args['mode']
    if mode == 'gan':
        mean = callback_args['transform']['mean']
        std = callback_args['transform']['std']
        total_samples = callback_args['total_samples']
        batch_size = callback_args['sample_size']
        classifier_model = get_model(callback_args['classifier_model_args']).to(device)
        transform = batch_normalize_transform(mean, std)
        return InceptionScoreCallback(classifier_model,
                                      outdir,
                                      batch_size=batch_size,
                                      total_samples=total_samples,
                                      transform=transform,
                                      mode=mode,
                                      device=device)
    else:
        raise NotImplementedError('generate_inception_metric_callback for classification is not implemented')


def generate_frechet_metric_callback(callback_args, device, outdir, dataset):
    mean = callback_args['transform']['mean']
    std = callback_args['transform']['std']
    total_samples = callback_args['total_samples']
    batch_size = callback_args['sample_size']
    classifier_model = get_model(callback_args['classifier_model_args'])
    classifier_model_layer = callback_args['classifier_model_layer']
    if classifier_model_layer:
        classifier_model = torch.nn.Sequential(*list(classifier_model.children())[:classifier_model_layer])
    classifier_model = classifier_model.to(device)
    transform = batch_normalize_transform(mean, std)
    return FrechetInceptionScoreCallback(outdir=outdir,
                                         classifier=classifier_model,
                                         batch_size=batch_size,
                                         total_samples=total_samples,
                                         transform=transform,
                                         device=device,
                                         dataset=dataset)


def generate_gan_embedding_sampler_callback(callback_args, device, outdir):
    return GanEmbeddingSampler(callback_args,
                               device,
                               outdir)


def generate_gan_sampler_callback(callback_args, device, outdir):
    return GanSampler(callback_args, device, outdir)


def generate_callbacks(arguments: dict,
                       dataset: BaseDataset,
                       device,
                       outdir) -> List[Callbacks]:
    callbacks = []
    """
    Create and add all callbacks instances to `callbacks` list 
    """
    for callback_arg in arguments['callbacks_args']:
        assert len(callback_arg.keys()) == 1
        key = list(callback_arg.keys())[0]
        if key == 'InceptionMetric':
            callbacks.append(generate_inception_metric_callback(callback_arg[key],
                                                                device,
                                                                outdir))
        elif key == 'GanSampler':
            callbacks.append(generate_gan_sampler_callback(callback_arg[key], device, outdir))
        elif key == 'FrechetMetric':
            callbacks.append(generate_frechet_metric_callback(callback_arg[key], device, outdir, dataset))
        elif key == 'GanEmbeddingSampler':
            callbacks.append(generate_gan_embedding_sampler_callback(callback_arg[key], device, outdir))

    return callbacks
