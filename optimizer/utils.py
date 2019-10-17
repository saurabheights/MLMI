import importlib


def create_optimizer(weights, kwargs: dict):
    optimizer_name = kwargs['name']
    kwargs = kwargs.copy()
    kwargs.pop('name')

    p, m = optimizer_name.rsplit('.', 1)  # p is module(filename), m is Class Name

    module_name = p
    module_obj = importlib.import_module(module_name)
    optimizer = getattr(module_obj, m)

    return optimizer(weights, **kwargs)
