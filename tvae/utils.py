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

def inverted_mexican_hat(weighted_distance_matrix):
    """
    Return matrix of lateral effects between neurons under Inverted Mexican Hat effect.

    Args:
        weighted_distance_matrix (ndarray): Matrix of sigma-weighted distances between neurons.

    Returns:
        ndarray: Matrix of lateral effect values between neurons.
    """
    S = -(1.0 - 0.5 * np.square(weighted_distance_matrix)) * np.exp(
        -0.5 * np.square(weighted_distance_matrix)
    )

    return S - np.eye(weighted_distance_matrix.shape[0])


def excitation_only(weighted_distance_matrix):
    """
    Return matrix of lateral effects between neurons under local excitatory effect.

    Args:
        weighted_distance_matrix (ndarray): Matrix of sigma-weighted distances between neurons.

    Returns:
        ndarray: Matrix of lateral effect values between neurons.
    """
    S = np.exp(-0.5 * np.square(weighted_distance_matrix))

    return S - np.eye(weighted_distance_matrix.shape[0])


def inhibition_only(weighted_distance_matrix):
    """
    Return matrix of lateral effects between neurons under local inhibitory effect.

    Args:
        weighted_distance_matrix (ndarray): Matrix of sigma-weighted distances between neurons.

    Returns:
        ndarray: Matrix of lateral effect values between neurons.
    """
    S = -(np.exp(-0.5 * np.square(weighted_distance_matrix)))

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

def pol2cart(r, theta):

    x = r * np.cos(np.radians(theta))
    y = r * np.sin(np.radians(theta))

    return np.array([x, y])

def plot_tuning_maps(tuning_map, idxs):
    """
    Plots tuning heatmaps for a set of neurons.

    Args:
        tuning_map (ndarray): array of predicted firing rates at different direction/speed combinations for a set of neurons.
        idxs (ndarray): indexes of neurons to plot.
    """
    fig, axs = plt.subplots(1, len(idxs), figsize=(18, 5))

    vmin, vmax = np.amin(tuning_map[:, idxs]), np.amax(tuning_map[:, idxs])
    for i, n in enumerate(idxs):
        im = axs[i].imshow(
            tuning_map[:, n].reshape(51, 360),
            aspect="auto",
            cmap="rainbow",
            origin="lower",
            vmin=vmin,
            vmax=vmax,
        )
        axs[i].set_ylim(0, 50)
        axs[i].set_xlim(0, 360)

    axs[0].set_ylabel("Speed (cm/s)", fontsize=25)
    axs[0].set_xlabel("Direction (°)", fontsize=25)
    axs[0].set_yticks([0, 25, 50])
    axs[0].set_xticks([0, 180, 360])
    axs[0].tick_params(labelsize=15)

    for i in range(1, len(idxs)):
        axs[i].set_xticks([])
        axs[i].set_yticks([])

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.show()

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


def generate_perturbation_input(natural_input):
    """
    Generate array of perturbation inputs, perturbing each input node over a range of values.

    Args:
        natural_input (ndarray): Array of natural movement inputs.

    Returns:
        array: Array of perturbation inputs.
    """

    perturbation_input = torch.vstack(
        [natural_input.mean(axis=0) for _ in range(natural_input.shape[1] * 100)]
    )

    for i in range(natural_input.shape[1]):
        mean = natural_input.mean(axis=0)[i]
        max_val = mean + natural_input.std(axis=0)[i] * 1
        min_val = mean - natural_input.std(axis=0)[i] * 1

        perturbation_input[i * 100 : i * 100 + 100, i] = torch.linspace(
            min_val, max_val, 100
        )

    return perturbation_input


def generate_latent_perturbations(model, perturbation_input):
    """
    Generate latent activities for a given model using perturbation inputs.

    Args:
        model (nn.module): TVAE model.
        perturbation_input (ndarray): Array of perturbation inputs.

    Returns:
        Array: Array of latent activities for each perturbation input.
    """

    model.cuda()
    with torch.no_grad():
        latent_perturbations = (
            model.encoder.forward(perturbation_input.cuda()).cpu().numpy()
        )
    model.cpu()

    return latent_perturbations


def plot_joint_correlations(latent_perturbations, perturbation_input):
    """
    Plot correlations of each neuron with each input node.

    Args:
        latent_perturbations (ndarray): Array of latent activities for each perturbation input.
        perturbation_input (ndarray): Array of perturbation inputs.
    """

    fig, axs = plt.subplots(3, 3, figsize=(10, 10))

    for x in range(3):
        for y in range(3):
            i = 3 * x + y
            corr_map = np.array(
                [
                    np.corrcoef(latent_perturbations[:, j], perturbation_input[:, i])[
                        0, 1
                    ]
                    for j in range(6400)
                ]
            ).reshape(80, 80)

            im = axs[x, y].imshow(corr_map, cmap="binary", vmin=-1, vmax=1)

            axs[x, y].set_yticks([])
            axs[x, y].set_xticks([])

    axs[0, 0].set_ylabel("Shoulder", fontsize=25)
    axs[1, 0].set_ylabel("Elbow", fontsize=25)
    axs[2, 0].set_ylabel("Wrist", fontsize=25)

    axs[2, 0].set_xlabel("X", fontsize=25)
    axs[2, 1].set_xlabel("Z", fontsize=25)
    axs[2, 2].set_xlabel("Y", fontsize=25)
    plt.show()


def joints2rgb(latent_perturbations, perturbation_input):
    """Convert joint correlations to a rgb colour code (red=Shoulder, Green=Elbow, Blue=Wrist)

    Args:
        latent_perturbations (ndarray): Array of latent activities for each perturbation input.
        perturbation_input (ndarray): Array of perturbation inputs.

    Returns:
        ndarray: Array of RGB values.
    """

    rgb = [
        np.stack(
            [
                np.array(
                    [
                        np.corrcoef(
                            latent_perturbations[:, j], perturbation_input[:, k + i]
                        )[0, 1]
                        for j in range(6400)
                    ]
                ).reshape(80, 80)
                for i in range(3)
            ]
        )
        for k in [0, 3, 6]
    ]
    rgb = np.stack([np.abs(x).sum(axis=0) for x in rgb])

    return rgb


def plot_joint_preference_map(rgb):
    """
    Plot joint preference RGB values on cortical map.

    Args:
        rgb (ndarray): Array of RGB values corresponding to joint preferences for each neuron.
    """
    rgb = rgb.reshape(3, -1)
    rgb = rgb.T / rgb.max(axis=1)
    rgb = rgb.T.reshape(3, 80, 80)
    swapped_rgb = np.swapaxes(np.swapaxes(rgb, 0, 1), 1, 2) / rgb.max()

    fig = plt.figure(figsize=(10, 10))
    plt.imshow(swapped_rgb, interpolation="nearest")
    plt.show()


def plot_3D_joint_preference_scatter(rgb):

    RGB = rgb.reshape(3, -1)
    RGB_c = RGB.T / RGB.max(axis=1)
    RGB = (RGB / RGB.sum(axis=0)) * 100

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection="3d")

    ax.view_init(15, 55)
    ax.scatter(RGB[1, :], RGB[0, :], RGB[2, :], c=RGB_c, alpha=0.5)

    ax.set_xlim([0, 100])
    ax.set_ylim([0, 100])
    ax.set_zlim([0, 100])
    ax.set_xlabel("Elbow (%)", fontsize=35, labelpad=20)
    ax.set_ylabel("Shoulder (%)", fontsize=35, labelpad=20)
    ax.set_zlabel("Wrist (%)", fontsize=35, labelpad=20)
    ax.tick_params(labelsize=20)
    plt.show()
