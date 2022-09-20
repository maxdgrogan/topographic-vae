import torch
from torch import nn


class BernoulliVAE(nn.Module):
    """
    Vanilla VAE with Bernoulli latent distribution.

    Args:
        input_size (int): size of model input
        latent_dim (list[int]): dimensions of 2D latent space
        n (int): number of times to sample from latent distribution for a single input
        tau (float): temperature parameter of gumbel-softmax function
        device (str): device to run inference on (cuda:0 or cpu)

    Methods:
        encode(x): encode an input
        decode(x): decode a sample from latent distribution
        uniform_sample(shape): sample tensor of size (shape) from U(0,1)
        gumbel_sample(u): sample tensor from gumbel distribution G(0,1) using U(0,1)
        sample(pi): sample continuous approximation of one hot vector from pi using gumbel-softmax function
        forward(x): encode and decode an input, also returning outputs from all intermediate steps
        KL_divergence(pi): calculate KL divergence of latent distribution from p(spike=0.5)
        round_counts(round_counts): make model return discrete spike counts (makes the forward-pass non-differentiable)
    """
    def __init__(self, input_size, latent_dim, n=1, tau=0.1, device="cuda:0") -> None:
        super(BernoulliVAE, self).__init__()

        self.device = device

        self.n = n  # Number times each neuron should be sampled
        self.tau = tau  # Temperature of Gumbel-Softmax

        # Decides if spike counts are rounded to integers in z (non-differentiable, so can't be done during training)
        self._round_counts = False  

        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 500, bias=False),
            nn.Tanh(),
            nn.Linear(500, latent_dim, bias=False),
            nn.Sigmoid(),
        )

        self.decoder = nn.Linear(latent_dim, input_size, bias=False)

    def encode(self, x):
        """
        encode an input

        Args:
            x (torch.tensor): input tensor with shape (batch_size, input_size)

        Returns:
            torch.tensor: spiking probabilities for latent neurons with size (batch_size, latent_dim[0] * latent_dim[1])
        """
        return self.encoder(x) * 0.99 + 0.005

    def decode(self, x):
        """
        decode a sample from the latent distribution

        Args:
            x (torch.tensor): spike counts from latent distribution sample with size (batch_size, latent_dim[0] * latent_dim[1])

        Returns:
            torch.tensor: reconstruction of original input with size (batch_size, input_size)
        """
        return self.decoder(x)

    def uniform_sample(self, shape):
        """sample a tensor of size (shape) from U(0,1)

        Args:
            shape (tuple[int]): dimensions of tensor to be sampled

        Returns:
            torch.tensor: sample of size (shape) from U(0,1)
        """
        return (
            torch.FloatTensor(*shape, self.n)
            .uniform_(0, 1 - 0.01 * self.tau)
            .to(self.device)
        )

    def gumbel_sample(self, u):
        """
        sample from gumbel distribution using sample from U(0,1)

        Args:
            u (torch.tensor): tensor of samples from U(0,1)

        Returns:
            torch.tensor: samples from G(0,1)
        """
        return -torch.log(-torch.log(u))

    def sample(self, pi):
        """
        sample from pi using gumbel-softmax function

        Args:
            pi (torch.tensor): tensor of spike and no-spike probabilities for latent neurons of size (2, batch_size, latent_dim[0] * latent_dim[1]) 

        Returns:
            torch.tensor: spike counts for latent neurons of size (batch_size, latent_dim[0] * latent_dim[1])
        """
        # Sample from U(0,1)
        u = self.uniform_sample((pi.shape[0], pi.shape[1], pi.shape[2]))

        # Use U(0,1) to sample from G(0,1)
        g = self.gumbel_sample(u)

        # Sample from pi
        y = (
            torch.exp((torch.log(pi) + g) / self.tau)
            / torch.exp((torch.log(pi) + g) / self.tau).sum(axis=0)[None, :, :, :]
        )

        # If no longer training, round to counts to int (at cost of being non-differentiable)
        return y.round() if self._round_counts else y

    def forward(self, x):
        """
        encode and decode an input, alsoreturning outputs from all intermediate steps

        Args:
            x (torch.tensor): input of size (batch_size, input_size)

        Returns:
            torch.tensor: reconstruction of input (xhat) (see decode() method)
            torch.tensor: spike count samples from latent distribution (z) (see sample method())
            torch.tensor: spike and no-spike probabilities for latent neurons (pi), derived from p (see encode() method)
            torch.tensor: firing probabilities for latent neurons (p) (see encode() method)
            torch.tensor: original input (x)
        """
        p = self.encode(x)
        pi = torch.stack([p, 1 - p])[:, :, :, None]  # Convert p to pi
        z = self.sample(pi)[:, :, :, :].sum(axis=3)[0, :, :]

        return self.decode(z), z, pi, p, x

    def KL_divergence(self, pi):
        """
        calculate KL divergence of latent distribution from p(spike=0.5)

        Args:
            pi (torch.tensor): spike and no-spike probabilities for latent neurons of size (2, batch_size, latent_dim[0] * latent_dim[1])

        Returns:
            torch.tensor: mean KL divergence of latent distribution
        """
        return (
            (pi * torch.log(pi / torch.full_like(pi, 0.5).to(self.device)))
            .sum(axis=1)
            .mean()
        )

    def round_counts(self, round_counts: bool):
        """
        make model return discrete spike counts (makes the forward-pass non-differentiable)

        Args:
            round_counts (bool): True or False
        """
        self._round_counts = round_counts


class BernoulliTopoVAE(BernoulliVAE):
    """
    bernoulli VAE with lateral effect 

    Args:
        psi (torch.tensor): tensor defining the desired pairwise interactions between neurons according to a chosen lateral effect function

    Methods:
        lateral_loss(z): calculate the lateral effect loss for a given sample from the latent distribution
    """
    def __init__(
        self, input_size, latent_dim, psi, n=1, tau=0.1, device="cuda:0"
    ) -> None:
        super().__init__(input_size, latent_dim, n, tau, device)

        self.psi = psi

    def lateral_loss(self, z):
        """
        calculate E[tr([Z'][Ψ][Z']^T)], the loss metric between current lateral effects and desired lateral effects

        Args:
            z (torch.tensor): tensor of spike counts sampled from latent distribution of size (batch_size, latent_dim[0] * latent_dim[1])

        Returns:
            torch.tensor : mean lateral effect loss
        """

        n = z.norm(2, 1).view(-1, 1).repeat(1, self.latent_dim)
        z = z / n

        A = z.mm(self.psi).mm(z.t()) / self.latent_dim
        return -torch.diag(A).mean()
