import importlib

import torch

import models


def convert_sample_to_batch(input):
    return input.unsqueeze(0)


def print_learnable_params(net: torch.nn.Module, recurse=True):
    params = list(net.parameters(), recurse)
    print(params)


def num_flat_features(x):
    """
    Computes number of features in a single sample of a batch
    :param x: The batch of inputs/features
    :return: Number of features in a single sample of a batch.
    """
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


def get_model(kwargs: dict) -> torch.nn.Module:
    """
    Returns module object for relative module name
    :param kwargs: dict containing model qualified name, weight path if available and model constructor params.
    :return: Module object for model name.
    """

    module_name, m = kwargs['model_arch_name'].rsplit('.', 1)  # p is module(filename), m is Class Name
    module_obj = importlib.import_module(module_name)
    model = getattr(module_obj, m)
    model: torch.nn.Module = model(**kwargs['model_constructor_args'])
    if kwargs['model_weights_path'] is not None:
        model.load_state_dict(torch.load(kwargs['model_weights_path']), strict=False)
    return model
