import datetime
import os

from utils import logger


def filter_optimizer_keys(value):
    if type(value) is str and 'torch.optim' in value:
        return value.rsplit(sep=".", maxsplit=1)[1]
    return value


def get_result_directory_name(timestamp,
                              mode,
                              model_arch_name: str,
                              generator_model_arch_name: str,
                              batch_size,
                              optimizer_args: dict,
                              discriminator_optimizer_args: dict,
                              generator_optimizer_args: dict,
                              dataset_args: dict):
    # Model and Dataset first
    dirname = f'{timestamp}_mode_{mode}'
    if mode == 'classification':
        dirname += f'_model_{model_arch_name.rsplit(sep=".", maxsplit=1)[1]}'
    else:
        dirname += f'_model_{generator_model_arch_name.rsplit(sep=".", maxsplit=2)[1]}'
    dirname += f'_dataset_{str(dataset_args["name"]).rsplit(".", maxsplit=1)[1].split("_")[0]}'
    dirname += f'_subset_{dataset_args["training_subset_percentage"]}'

    # Training Hyperparams - Batch Size, Optimizer
    dirname += f'_bs_{batch_size}'

    if mode == 'classification':
        for key, value in optimizer_args.items():
            dirname += f'_{key}_{filter_optimizer_keys(value)}'
    else:
        dirname += '_G'
        for key, value in generator_optimizer_args.items():
            dirname += f'_{key}_{filter_optimizer_keys(value)}'
        dirname += '_D'
        for key, value in discriminator_optimizer_args.items():
            dirname += f'_{key}_{filter_optimizer_keys(value)}'

    return dirname


def make_results_dir(arguments):
    """
    Create a timestamped output directory with training details
    :return: The output directory path.
    """
    mode = arguments['mode']
    optimizer_args = arguments.get('optimizer_args', None)
    generator_optimizer_args = arguments.get('generator_optimizer_args', None)
    discriminator_optimizer_args = arguments.get('discriminator_optimizer_args', None)

    outdir = arguments.get("outdir")
    model_arch_name = arguments.get('model_args', dict()).get('model_arch_name', None)
    generator_model_arch_name = arguments.get('generator_model_args', dict()).get('model_arch_name', None)
    discriminator_model_arch_name = arguments.get('discriminator_model_args', dict()).get('model_arch_name', None)

    train_data_args = arguments.get('train_data_args')
    batch_size = train_data_args.get("batch_size", 1)

    dirname = get_result_directory_name(timestamp=datetime.datetime.now().isoformat(),
                                        mode=mode,
                                        batch_size=batch_size,
                                        model_arch_name=model_arch_name,
                                        generator_model_arch_name=generator_model_arch_name,
                                        optimizer_args=optimizer_args,
                                        generator_optimizer_args=generator_optimizer_args,
                                        discriminator_optimizer_args=discriminator_optimizer_args,
                                        dataset_args=arguments['dataset_args'])

    outdir = os.path.join(outdir, dirname)
    os.makedirs(outdir, exist_ok=True)
    logger.info("Output directory: %s", outdir)
    return outdir


def make_test_results_dir(arguments):
    outdir = arguments.get("outdir")
    dirname = datetime.datetime.now().isoformat() + '_Test'
    outdir = os.path.join(outdir, dirname)
    os.makedirs(outdir, exist_ok=True)
    logger.info("Output directory: %s", outdir)
    return outdir


def delete_old_file(path):
    os.remove(path)
