import os

import matplotlib.pyplot as plt
import numpy
import torch
import torchvision
from tqdm import tqdm

from callbacks.callbacks import Callbacks
from models.evaluation.inception import InceptionV3
from utils import logger
from utils.progress_bar import ProgressBar
from utils.tensorboard_writer import initialize_tensorboard


class GanEmbeddingSampler(Callbacks):
    """
    This callback plots embeddings of Gan Images from Inception Network in tensorboard.
    """
    def __init__(self,
                 gan_embedding_sampler_args: dict,
                 device,
                 outdir):
        super(GanEmbeddingSampler, self).__init__()
        self.device = device
        self.outdir = outdir
        self.gan_embedding_sampler_args = gan_embedding_sampler_args.copy()
        self.num_samples = self.gan_embedding_sampler_args['num_samples']
        self.step_size = self.gan_embedding_sampler_args.get('step_size', 1)
        self.batch_size = self.gan_embedding_sampler_args.get('batch_size', 8)

        self.run_on_nth_iteration = gan_embedding_sampler_args.get('run_on_nth_iteration', False)
        self.run_on_train_end = gan_embedding_sampler_args.get('run_on_train_end', True)

        # To always display one saliency map. Allows quick catching of error.
        self.debug = True

    def on_nth_iteration(self, iteration):
        if self.run_on_nth_iteration:
            self.generate_generator_embeddings(iteration, tag='TrainInceptionEmbedding')

    def on_train_end(self):
        self.generate_generator_embeddings(0, tag='EvalInceptionEmbedding')

    def generate_generator_embeddings(self, iteration, tag):
        """
        Embedding of self.num_samples samples are written to Tensorboard.
        :return: None
        """
        tensorboard_writer, iteration_output_dir = None, None
        if self.gan_embedding_sampler_args['write_to_tensorboard']:
            tensorboard_writer = initialize_tensorboard(self.outdir)

        # ToDo: if batch size is smaller than num_samples
        num_steps = (self.num_samples + self.batch_size - 1) // self.batch_size
        total_embeddings = None
        total_images = None
        for _ in tqdm(range(num_steps)):
            generator = self.model[0]
            generator.eval()
            with torch.no_grad():
                noise = torch.randn(self.batch_size, generator.z_dim, 1, 1, device=self.device)
                images = generator(noise).detach()
                images = images.repeat(1, 3, 1, 1)
                images = images.cpu()
                embeddings = generator.embeddings.reshape((generator.embeddings.shape[0], -1)).cpu()
                if total_embeddings is not None:
                    total_embeddings = torch.cat((total_embeddings, embeddings), 0)
                    total_images = torch.cat((total_images, images), 0)
                else:
                    total_embeddings = embeddings
                    total_images = images

        tensorboard_writer.save_embedding(mat=total_embeddings,
                                          label_img=total_images,
                                          global_step=iteration,
                                          tag=tag)
