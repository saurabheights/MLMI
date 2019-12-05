import torch
import torch.nn


class Generator(torch.nn.Module):
    def __init__(self, z_dim, latent_dim, image_size):
        super().__init__()
        self.z_dim = z_dim
        X_dim = image_size[0] * image_size[1] * image_size[2]
        # Create the Generator
        self.G = torch.nn.Sequential(
            torch.nn.Linear(z_dim, latent_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(latent_dim, X_dim),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.G(x)


class Discriminator(torch.nn.Module):
    def __init__(self, latent_dim, image_size):
        super().__init__()
        X_dim = image_size[0] * image_size[1] * image_size[2]
        # Create the discriminator
        self.D = torch.nn.Sequential(
            torch.nn.Linear(X_dim, latent_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(latent_dim, 1),
        )

    def forward(self, x):
        x = x.reshape((x.shape[0], -1))
        return self.D(x)