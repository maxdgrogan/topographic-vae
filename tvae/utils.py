import numpy as np
import torch
import h5py
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
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


def load_natural_data(path, include_angles=False, include_velocities=True):
    """
    Load and preprocess natural arm kinematic data.

    Args:
        path (str): Path to folder with kinematic data files (hdf5 formatted).
        include_angles (bool, optional): Whether to include joint angle data. Defaults to True.
        include_velocities (bool, optional): Whether to include joint angle velocity data. Defaults to True.

    Raises:
        ValueError: If neither joint angles are joint angle velocities are specified to be included.

    Returns:
        torch.tensor: torch tensor of preprocessed arm kinematic data for natural tasks.
    """

    i = [21, 22, 23, 24, 25, 26, 27, 28, 29]
    all_data = []

    if not include_angles and not include_velocities:
        raise ValueError("both angles and velocities are False.")

    for file in os.listdir(path):
        if file[-3:] == ".h5":

            f = h5py.File(path + file, "r")
            angles = f["data"][:, i]
            # angles = f["data"][:]
            velocities = np.gradient(angles, axis=0)
            f.close()

            if include_angles and include_velocities:
                data = np.hstack([angles, velocities])
            else:
                data = angles if include_angles else velocities

            all_data.append(data)

    all_data = np.vstack(all_data)

    return torch.tensor(all_data).float()

def load_planar_data(path, include_angles=False, include_velocities=True):
    """
    Load and preprocess planar arm kinematic data.

    Args:
        path (str): Path to folder with kinematic data files (hdf5 formatted).
        include_angles (bool, optional): Whether to include joint angle data. Defaults to True.
        include_velocities (bool, optional): Whether to include joint angle velocity data. Defaults to True.

    Raises:
        ValueError: If neither joint angles are joint angle velocities are specified to be included.

    Returns:
        torch.tensor: torch tensor of preprocessed arm kinematic data for planar reaching task.
    """

    with h5py.File(path) as file:
        angles = file["jointangle"][:]
        velocities = file["jointvelocity"][:]

    # indexes for centre out reaching movements
    centre_out_idxs = np.hstack(
        [range(1000, 11000), range(16000, 21000), range(26000, 27000)]
    )
    angles = angles[centre_out_idxs, :]
    velocities = velocities[centre_out_idxs, :]

    if not include_angles and not include_velocities:
        raise ValueError("both angles and velocities are False.")


    if include_angles and include_velocities:
        data = np.hstack([angles, velocities])
    else:
        data = angles if include_angles else velocities

    return torch.tensor(data).float()

def load_kinematic_data(path, include_angles=False, include_velocities=True, seed=0):
    """
    Load and preprocess natural and planar arm kinematic data.

    Args:
        path (str): Path to folder with kinematic data files (hdf5 formatted).
        include_angles (bool, optional): Whether to include joint angle data. Defaults to True.
        include_velocities (bool, optional): Whether to include joint angle velocity data. Defaults to True.

    Raises:
        ValueError: If neither joint angles are joint angle velocities are specified to be included.

    Returns:
        List[torch.tensor]: torch tensors of preprocessed arm kinematic data (natural, planar).
    """
        
    # Load natural kinematic data
    natural_data = load_natural_data(
        path + "natural/", include_angles=include_angles, include_velocities=include_velocities
    )

    # Load planar kinematic data
    planar_data = load_planar_data(
        path + "planar/planar_data.h5",
        include_angles=include_angles, include_velocities=include_velocities
    )

    # Subsample at 1%
    np.random.seed(seed)
    n = int(0.01 * natural_data.shape[0])
    idxs = np.linspace(0, natural_data.shape[0] - 1, n).astype("int")
    natural_data = natural_data[idxs, :]

    # Standardise variance
    stds = torch.vstack([natural_data, planar_data]).std(axis=0)
    return natural_data / stds, planar_data / stds

def plot_PD_map(PDs):
    """
    Plot cortical map of preferred directions for latent space of model.

    Args:
        PDs (ndarray): Array of preferred directions in cortical map.
    """

    colors = (
        np.array(
            [
                [239, 137, 71],
                [254, 220, 87],
                [116, 212, 123],
                [100, 157, 236],
                [109, 132, 228],
                [174, 93, 224],
                [217, 91, 226],
                [233, 94, 97],
            ]
        )
        / 255
    )
    cmap = LinearSegmentedColormap.from_list("Custom map", colors, 8)

    fig = plt.figure(figsize=(8, 8))
    plt.imshow(np.array(PDs).reshape(80, 80), cmap=cmap)
    plt.clim([-22.5, 337.5])
    plt.colorbar(ticks=np.linspace(0, 315, 8))
