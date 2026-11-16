import numpy as np
from numpy.fft import fftn, ifftn, fftfreq
import yaml
import torch
from numpy import fft

from pathlib import Path
from types import SimpleNamespace


def compute_omni_spectrum_2D(velocity_field, dx=1.0, dy=1.0):
    # Get dimensions
    n, _, nx, ny = velocity_field.shape

    # Compute mean velocity field
    mean_velocity = np.mean(velocity_field, axis=0)
    fluctuations = velocity_field - mean_velocity[np.newaxis, :, :, :]

    # Calculate wavenumbers
    kx = 2 * np.pi * fft.fftfreq(nx, dx)
    ky = 2 * np.pi * fft.fftfreq(ny, dy)

    # Create 2D wavenumber grid
    kx_2d, ky_2d = np.meshgrid(kx, ky, indexing='ij')
    k_mag = np.sqrt(kx_2d**2 + ky_2d**2)

    # Define wavenumber bins
    dk = min(2*np.pi/(nx*dx), 2*np.pi/(ny*dy))
    k_max = np.sqrt(2) * max(np.max(np.abs(kx)), np.max(np.abs(ky)))
    k_bins = np.arange(dk/2, k_max, dk)

    # Initialize energy spectra for each component
    Ek = np.zeros((3, len(k_bins)))
    modes_in_shell = np.zeros_like(k_bins)

    # Compute spectrum for each timestep and each component separately
    for t in range(n):
        for i in range(3):  # Loop over u, v, w components
            # 2D FFT of velocity component
            u_hat = fft.fft2(fluctuations[t, i])

            # Energy density in Fourier space
            energy_density = np.abs(u_hat)**2

            # Bin the energy by wavenumber magnitude
            for ki in range(len(k_bins)):
                k_low = k_bins[ki] - dk/2
                k_high = k_bins[ki] + dk/2
                shell = (k_mag >= k_low) & (k_mag < k_high)

                Ek[i, ki] += np.sum(energy_density[shell])
                if i == 0:  # Only count modes once
                    modes_in_shell[ki] += np.sum(shell)

    # Average over samples
    Ek /= n

    # Normalize by number of modes in each shell
    valid_bins = modes_in_shell > 0
    Ek[:, valid_bins] /= modes_in_shell[valid_bins]

    # Apply normalization factor for energy conservation
    Ek *= (dx * dy) / (nx * ny)

    return k_bins, Ek


class Config:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Convert nested dictionaries to SimpleNamespace objects
        self.config = self._dict_to_namespace(config_dict)


    def _dict_to_namespace(self, d):
        """Recursively convert dictionary to SimpleNamespace"""
        for key, value in d.items():
            if isinstance(value, dict):
                d[key] = self._dict_to_namespace(value)
            elif isinstance(value, list):
                # Handle lists of dictionaries (e.g., train_cases)
                d[key] = [self._dict_to_namespace(item) if isinstance(item, dict) else item
                         for item in value]
        return SimpleNamespace(**d)


    def __getattr__(self, name):
        return getattr(self.config, name)

def load_config(config_path):
    """Load configuration from YAML file"""
    return Config(config_path)


def get_vel_vector(data_path,num_snapshot,dimensions):
    """Load velocity components and stack them."""
    dtype=np.float64

    u = np.fromfile(data_path + rf'ux-{num_snapshot}.bin',
                    dtype=dtype).reshape(dimensions, order='F')
    v = np.fromfile(data_path + rf'uy-{num_snapshot}.bin',
                    dtype=dtype).reshape(dimensions, order='F')
    w = np.fromfile(data_path + rf'uz-{num_snapshot}.bin',
                    dtype=dtype).reshape(dimensions, order='F')

    # Trim the last column
    u = u[:, 1:, :]
    v = v[:, 1:, :]
    w = w[:, 1:, :]

    velocity_field = np.stack((u, v, w), axis=0)
    return velocity_field


def normalize_channel(tensor):
    """Normalize tensor to [0, 1] and return normalization parameters."""
    normalized = tensor.clone()
    min_vals = torch.zeros(3, device=tensor.device)  # Store min values per channel
    max_vals = torch.zeros(3, device=tensor.device)  # Store max values per channel

    for c in range(3):
        min_vals[c] = tensor[c].min()
        max_vals[c] = tensor[c].max()
        normalized[c] = (tensor[c] - min_vals[c]) / (max_vals[c] - min_vals[c])  # Normalize

    return normalized, min_vals, max_vals


def denormalize_channel(normalized, min_vals, max_vals):
    """Reverses normalization using stored min/max values before saving."""
    device = normalized.device  # Ensure all tensors are on the same device
    min_vals = min_vals.to(device)
    max_vals = max_vals.to(device)

    denormalized = normalized.clone()
    for c in range(3):
        denormalized[c] = normalized[c] * (max_vals[c] - min_vals[c]) + min_vals[c]

    return denormalized
