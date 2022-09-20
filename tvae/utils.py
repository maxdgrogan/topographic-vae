import numpy as np
from sklearn.metrics import euclidean_distances


def mexican_hat(weighted_distance_matrix):
    """
    Return matrix of lateral effects between neurons under Mexican Hat effect.

    Args:
        weighted_distance_matrix (ndarray): Matrix of sigma-weighted distances between neurons.

    Returns:
        ndarray: Matrix of lateral effect values between neurons.
    """
    S = (1.0 - 0.5 * np.square(weighted_distance_matrix)) * np.exp(
        -0.5 * np.square(weighted_distance_matrix)
    )

    return S - np.eye(weighted_distance_matrix.shape[0])


def locmap(latent_shape):
    """
    Generate array of 2D positions for each neuron in latent space.

    Args:
        latent_shape (ndarray): Array with shape (,2), specifying height and width of latent space.

    Returns:
        ndarray: Array with shape (M*N, 2), containing 2D position for each neuron in latent space.
    """

    x = np.arange(0, latent_shape[0], dtype=np.float32)
    y = np.arange(0, latent_shape[1], dtype=np.float32)
    xv, yv = np.meshgrid(x, y)
    xv = np.reshape(xv, (xv.size, 1))
    yv = np.reshape(yv, (yv.size, 1))

    return np.hstack((xv, yv))


def lateral_effect(latent_shape, sigma=3, f=mexican_hat):
    """
    Create a matrix defining pairwise lateral effects between neurons in a 2D latent space of specified size.

    Args:
        latent_shape (ndarray): Array with shape (,2), specifying height and width of latent space.
        sigma (int, optional): Lateral effect range. Defaults to 3.

    Returns:
        ndarray: Array with shape (M*N, M*N), defining the values of lateral effects between neurons.
    """

    locations = locmap(latent_shape)
    weighted_distance_matrix = euclidean_distances(locations, locations) / sigma

    return f(weighted_distance_matrix)
