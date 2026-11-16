import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
import torch.nn.functional as F
import math
import random
import time
from pathlib import Path
import pickle
class abl_Dataset(Dataset):

    def __init__(self,config):

        self.data_paths = config.data.paths_hr
        self.data_paths_LR = config.data.paths_lr

        self.z0s = config.data.z0s
        self.Ugs = config.data.Ugs
        self.normalization = config.data.normalization

        print(len(self.data_paths))
        self.dimensions = config.data.dimensions
        self.condition_kernel = config.data.condition_kernel


        #self.snapshot_indices = list(range(cfd_case.start_idx, cfd_case.end_idx + 1))
        self.snapshot_indices = list(range(config.data.start_idx, config.data.end_idx + 1))
        #self.snapshot_indices = list(range(110, 195 + 1))
        self.data_type = np.float64
        self.scale_factor = config.data.scale_factor

        if self.scale_factor == 8:
            self.coarse_grid_number = 0
        elif self.scale_factor == 4:
            self.coarse_grid_number = 1
        elif self.scale_factor == 2:
            self.coarse_grid_number = 2

        # Calculate dimensions
        self.n_y_slices = self.dimensions[1]-1  # Number of z slices
        self.total_slices = len(self.snapshot_indices) * self.n_y_slices # * len(self.data_paths)
        self.hr_height = self.dimensions[0]
        self.hr_width = self.dimensions[2] - 1  # Trim last column

        self.lr_dimensions = [self.dimensions[0]//self.scale_factor,self.dimensions[1]//self.scale_factor +1,self.dimensions[2]//self.scale_factor]

        self.lr_height = self.hr_height // self.scale_factor
        self.lr_width = self.hr_width // self.scale_factor
        self.lr_n_y_slices = self.n_y_slices//self.scale_factor


        ## Load stats data
        with open("datasets_stats/abl_stats.pkl", "rb") as f:
            self.stats_dict = pickle.load(f)


        #print(f"Data path: {self.data_path}")
        #print(f"Snapshots: {min(self.snapshot_indices)} to {max(self.snapshot_indices)}")
        #print(f"Dimensions - HR: {self.hr_height}x{self.hr_width}, LR: {self.lr_height}x{self.lr_width}")

        #self._validate_data_files()
    def __len__(self):
        """Ensure correct dataset size considering all paths and slices."""
        return len(self.snapshot_indices) * self.n_y_slices * len(self.data_paths)

    def _validate_data_files(self):
        """Check if all required binary files exist."""
        first_idx = self.snapshot_indices[0]
        test_file = self.data_path / f'ux-{first_idx}.bin'
        print(f"Testing file existence: {test_file}")
        if not test_file.exists():
            raise FileNotFoundError(f"Cannot find first test file: {test_file}")

        missing_files = []
        for idx in self.snapshot_indices:
            for component in ['ux', 'uy', 'uz']:
                filepath = self.data_path / f'{component}-{idx}.bin'
                if not filepath.exists():
                    missing_files.append(str(filepath))

        if missing_files:
            raise FileNotFoundError(f"Missing data files: {missing_files}")
        print("All data files found successfully!")


########################
    def get_vel_vector(self, data_path , dimension,  num_snapshot):
        """Load velocity components and stack them."""
        try:

          #  print(dimension)
            u = np.fromfile(data_path/f'ux-{num_snapshot}.bin',
                           dtype=self.data_type).reshape(dimension, order='F')
            v = np.fromfile(data_path/f'uy-{num_snapshot}.bin',
                           dtype=self.data_type).reshape(dimension, order='F')
            w = np.fromfile(data_path/f'uz-{num_snapshot}.bin',
                           dtype=self.data_type).reshape(dimension, order='F')

            # Trim the last column
            u = u[:, 1:, :]
            v = v[:, 1:, :]
            w = w[:, 1:, :]


            velocity_field = np.stack((u, v, w), axis=0)
            # Apply shuffle for the last axis
            #shuffled_indices = np.random.permutation(velocity_field.shape[-1])
            # Apply the shuffled indices along the last axis
            #velocity_field = velocity_field[..., shuffled_indices]

            return velocity_field

        except Exception as e:
            print(f"Error loading snapshot {num_snapshot}: {str(e)}")
            raise

    def _get_slice(self, velocity_field, y_idx):
        """Extract a slice from the velocity field."""
        slice_data = velocity_field[:, :,y_idx ,: ].copy()

      #  if slice_data.shape != (3, self.hr_height, self.hr_width):
          #  raise ValueError(f"Unexpected slice shape: {slice_data.shape}, expected {(3, self.hr_height, self.hr_width)}")

        return slice_data


    def _index_to_snapshot_and_slice(self, idx,n_y_slices):
        """Convert dataset index to snapshot index and z-slice index."""
        snapshot_idx = self.snapshot_indices[idx // n_y_slices]
        y_idx = idx % n_y_slices
        return snapshot_idx, y_idx


    def normalize_channel_mean_std(self,tensor,stats):
        for c in range(3):
            # Subtract mean
            tensor[c] = (tensor[c] - stats['mean'][c])/ stats['std'][c]
        return tensor


    def __getitem__(self, idx):
        """Efficiently get a pair of LR and HR slices with physical parameters."""

        try:

            # Select normalization function
            normalization_function = {
                'mean_std': self.normalize_channel_mean_std,
                #'zero_max': self.normalize_channel_zero_max,
                #'neg_pos': self.normalize_channel_neg_pos
            }[self.normalization]


            dataset_index = idx % len(self.data_paths)
            z0, Ug = self.z0s[dataset_index], self.Ugs[dataset_index]

            actual_idx = idx // len(self.data_paths)

            # Index computation
            snapshot_idx, y_idx = self._index_to_snapshot_and_slice(actual_idx, self.n_y_slices)
            snapshot_idx_lr, y_idx_lr = self._index_to_snapshot_and_slice(actual_idx // self.scale_factor, self.lr_n_y_slices)

            # Load HR Data
            velocity_field_hr = self.get_vel_vector(Path(self.data_paths[dataset_index]), self.dimensions, snapshot_idx)
            #print(velocity_field_hr.shape)

            stats = self.stats_dict[Ug][z0][self.coarse_grid_number]
            #velocity_field_hr = normalization_function(velocity_field_hr,stats)  # Only normalize once


            hr_slice = torch.from_numpy(self._get_slice(velocity_field_hr, y_idx)).float()
            hr_slice = normalization_function(hr_slice,stats)


            # Smooth LR
            smooth_lr_slice = self._get_slice(velocity_field_hr, y_idx)
            smooth_lr_slice = normalization_function(smooth_lr_slice,stats)
            smooth_lr_tensor = torch.from_numpy(smooth_lr_slice).float()
            #smooth_lr_tensor = F.avg_pool2d(smooth_lr_tensor, kernel_size=self.scale_factor**2, stride=self.scale_factor**2)
            smooth_lr_tensor = F.avg_pool2d(smooth_lr_tensor, kernel_size=self.condition_kernel, stride=self.condition_kernel)

          #  smooth_lr_tensor = F.interpolate(smooth_lr_tensor.unsqueeze(0),
           #                                 size=(hr_slice.shape[-2], hr_slice.shape[-1]),
            #                                mode='nearest-exact').squeeze(0)

            # Load LR Data
            velocity_field_real_lr = self.get_vel_vector(Path(self.data_paths_LR[dataset_index]), self.lr_dimensions, snapshot_idx_lr)
            #velocity_field_real_lr = normalization_function(velocity_field_real_lr,stats)
            real_lr_slice = self._get_slice(velocity_field_real_lr, y_idx_lr)
            real_lr_slice = normalization_function(real_lr_slice,stats)
            real_lr_tensor = torch.from_numpy(real_lr_slice).float()
            #real_lr_tensor = F.avg_pool2d(real_lr_tensor, kernel_size=self.scale_factor, stride=self.scale_factor)

            real_lr_tensor = F.avg_pool2d(real_lr_tensor, kernel_size=int(self.condition_kernel/self.scale_factor), stride= int(self.condition_kernel/self.scale_factor))

           # real_lr_tensor = F.interpolate(real_lr_tensor.unsqueeze(0),
            #                            size=(hr_slice.shape[-2], hr_slice.shape[-1]),
             #                           mode='nearest-exact').squeeze(0)
            return hr_slice, smooth_lr_tensor, real_lr_tensor, z0, Ug


        except Exception as e:
            print(f"Error loading data at idx {idx}: {e}")
            raise
