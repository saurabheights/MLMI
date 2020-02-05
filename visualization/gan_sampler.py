import os

import matplotlib.pyplot as plt
import numpy
import torch
import torchvision

from callbacks.callbacks import Callbacks
from utils import logger
from utils.progress_bar import ProgressBar
from utils.tensorboard_writer import initialize_tensorboard


class GanSampler(Callbacks):
    def __init__(self,
                 gan_sampler_args: dict,
                 device,
                 outdir,
                 debug=True):
        super(GanSampler, self).__init__()
        self.device = device
        self.outdir = outdir
        self.gan_sampler_args = gan_sampler_args.copy()
        self.num_samples = self.gan_sampler_args['num_samples']

        # To always display one saliency map. Allows quick catching of error.
        self.debug = True

    def on_nth_iteration(self, iteration):
        self.generate_generator_samples(iteration)

    def generate_generator_samples(self, iteration):
        """
        N fake sample images are written to Tensorboard.
        :return: None
        """
        tensorboard_writer, iteration_output_dir = None, None
        if self.gan_sampler_args['write_to_tensorboard']:
            tensorboard_writer = initialize_tensorboard(self.outdir)

        # ToDo: if batch size is smaller than num_samples
        generator = self.model[0]
        generator.eval()
        noise = torch.randn(self.num_samples, generator.nz, 1, 1, device=self.device)
        with torch.no_grad():
            images = generator(noise).detach()
            images = images.cpu()
            images = images.reshape(-1, 3, 64, 64)
            images = images.mul(0.5).add(0.5)
            image = torchvision.utils.make_grid(images, nrow=4)  # ToDo Remove hardcoded value

        # Convert 1-channel image to 3-channel.
        if image.dim() == 3 and image.shape[0] == 1:
            image = torch.squeeze(image)
            image = torch.stack((image,) * 3, dim=0)

        # Convert Channel First to Channel Last for matplotlib
        image = image.permute(1, 2, 0)
        image = image.numpy()

        # GrayScale Images are colormapped by default
        # Colored Images are not colormapped. 3-channel float Images however are getting warped around 0-1
        # range due to cast to np.uint8 without checking for minimum and maximum values in saliency.
        # https://github.com/matplotlib/matplotlib/issues/9391/
        fig = plt.figure(figsize=(8, 8), facecolor='w')
        plt.subplot(1, 1, 1)
        plt.title("Generated Fake Images")
        plt.imshow(image, cmap=None)
        plt.axis('off')

        if self.gan_sampler_args['write_to_tensorboard']:
            # https://github.com/lanpa/tensorboardX/issues/152
            tag = f"GeneratedImages_Iter_{iteration}"
            tensorboard_writer.save_figure(tag, fig, global_step=iteration)
        if self.gan_sampler_args['write_to_disk']:
            plt.savefig(f'{self.outdir}/{iteration}.jpg')

        # if self.debug:
        #     plt.show()
        # else:
        #     plt.close(fig)
