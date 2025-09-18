# import warnings
# warnings.filterwarnings("ignore")

import random
import numpy as np
import os.path as osp
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import tqdm

from utils.utils import create_loader

try:
    import xarray as xr
except ImportError:
    xr = None

d2r = np.pi / 180


def latlon2xyz(lat, lon):
    if type(lat) == torch.Tensor:
        x = -torch.cos(lat)*torch.cos(lon)
        y = -torch.cos(lat)*torch.sin(lon)
        z = torch.sin(lat)

    if type(lat) == np.ndarray:
        x = -np.cos(lat)*np.cos(lon)
        y = -np.cos(lat)*np.sin(lon)
        z = np.sin(lat)
    return x, y, z


def xyz2latlon(x, y, z):
    if type(x) == torch.Tensor:
        lat = torch.arcsin(z)
        lon = torch.atan2(-y, -x)

    if type(x) == np.ndarray:
        lat = np.arcsin(z)
        lon = np.arctan2(-y, -x)
    return lat, lon


data_map = {
    'z': 'geopotential',
    't': 'temperature',
    'tp': 'total_precipitation',
    't2m': '2m_temperature',
    'r': 'relative_humidity',
    's': 'specific_humidity',
    'u10': '10m_u_component_of_wind',
    'u': 'u_component_of_wind',
    'v10': '10m_v_component_of_wind',
    'v': 'v_component_of_wind',
    'tcc': 'total_cloud_cover',
    "lsm": "constants",
    "o": "constants",
    "l": "constants",
}

mv_data_map = {
    **dict.fromkeys(['mv', 'mv4'], ['r', 't', 'u', 'v']),
    'mv5': ['z', 'r', 't', 'u', 'v'],
    'uv10': ['u10', 'v10'],
    'mv12': ['lsm', 'o', 't2m', 'u10', 'v10', 'l', 'z', 'u', 'v', 't', 'r', 's']
}

data_keys_map = {
    'o': 'orography',
    'l': 'lat2d',
    's': 'q'
}


class WeatherBenchDataset(Dataset):
    """Wheather Bench Dataset <http://arxiv.org/abs/2002.00469>`_

    Args:
        data_root (str): Path to the dataset.
        data_name (str|list): Name(s) of the weather modality in Wheather Bench.
        training_time (list): The arrange of years for training.
        idx_in (list): The list of input indices.
        idx_out (list): The list of output indices to predict.
        step (int): Sampling step in the time dimension.
        levels (int|list|"all"): Level(s) to use.
        data_split (str): The resolution (degree) of Wheather Bench splits.
        use_augment (bool): Whether to use augmentations (defaults to False).
    """

    def __init__(self, data_root, data_name, training_time,
                 idx_in, idx_out, step=1, levels=['50'], data_split='5_625',
                 mean=None, std=None,
                 transform_data=None, transform_labels=None, use_augment=False):
        super().__init__()
        self.data_root = data_root
        self.data_split = data_split
        self.data_name = data_name
        self.training_time = training_time
        self.idx_in = np.array(idx_in)
        self.idx_out = np.array(idx_out)
        self.step = step
        self.data = None
        self.mean = mean
        self.std = std
        self.transform_data = transform_data
        self.transform_labels = transform_labels
        self.use_augment = use_augment

        self.time = None
        self.time_size = self.training_time

        self.data, self.mean, self.std = [], [], []

        if levels == 'all':
            levels = ['50', '250', '500', '600', '700', '850', '925']
        levels = levels if isinstance(levels, list) else [levels]
        levels = [int(level) for level in levels]
        if isinstance(data_name, str) and data_name in mv_data_map:
            data_names = mv_data_map[data_name]
        else:
            data_names = data_name if isinstance(data_name, list) else [data_name]

        for name in tqdm.tqdm(data_names):
            data, mean, std = self._load_data_xarray(data_name=name, levels=levels)
            self.data.append(data)
            self.mean.append(mean)
            self.std.append(std)

        # for i, data in enumerate(self.data):
        #     if data.shape[0] != self.time_size:
        #         self.data[i] = data.repeat(self.time_size, axis=0)

        self.data = np.concatenate(self.data, axis=1)
        self.mean = np.concatenate(self.mean, axis=1)
        self.std = np.concatenate(self.std, axis=1)

        # Robust valid index computation so that i+offsets stay within [0, T-1]
        min_offset = int(min(np.min(self.idx_in), np.min(self.idx_out)))
        max_offset = int(max(np.max(self.idx_in), np.max(self.idx_out)))
        start = 0 if min_offset >= 0 else -min_offset
        end = int(self.data.shape[0]) - (max_offset if max_offset > 0 else 0)
        end = max(start, end)
        self.valid_idx = np.arange(start, end)

    def _load_data_xarray(self, data_name, levels):
        """Loading full data with xarray (.nc directory, single .nc, or single .zarr)."""
        try:
            if isinstance(self.data_root, str):
                if self.data_root.endswith('.zarr'):
                    dataset = xr.open_zarr(self.data_root)
                elif self.data_root.endswith('.nc'):
                    dataset = xr.open_dataset(self.data_root)
                else:
                    # directory: try subdir layout .../<var>/<var>*.nc, else flat .../*<var>*.nc
                    subdir = osp.join(self.data_root, data_map[data_name])
                    pattern = None
                    if osp.isdir(subdir):
                        pattern = osp.join(subdir, f"{data_map[data_name]}*.nc")
                    else:
                        pattern = osp.join(self.data_root, f"*{data_map[data_name]}*.nc")
                    dataset = xr.open_mfdataset(pattern, combine='by_coords')
            else:
                pattern = self.data_root+'/{}/{}*.nc'.format(data_map[data_name], data_map[data_name])
                dataset = xr.open_mfdataset(pattern, combine='by_coords')
        except (AttributeError, ValueError):
            assert False and 'Please install xarray and its dependency (e.g., netcdf4), ' \
                                'pip install xarray==0.19.0,' \
                                'pip install netcdf4 h5netcdf dask'
        except OSError:
            print("OSError: Invalid path {}/{}/*.nc".format(self.data_root, data_map[data_name]))
            assert False

        # ###########
        # dataset = dataset.isel(time=slice(0, 100))
        # ###########

        if 'time' not in dataset.indexes:
            dataset = dataset.expand_dims(dim={"time": 1}, axis=0)
        else:
            # Respect training_time only if it's a range; allow scalar or single value
            if isinstance(self.training_time, (list, tuple)) and len(self.training_time) == 2:
                dataset = dataset.sel(time=slice(*self.training_time))
            # # reduce to per-frame (single) timestep
            # dataset = dataset.isel(time=slice(0, 1))
            # self.time_size = 1
        lat = dataset["latitude"] if "latitude" in dataset.coords else dataset["lat"]
        # 위도가 내림차순(90→-90)이면 한 방에 뒤집기
        if np.all(np.diff(lat.values) < 0):
            dataset = dataset.isel({lat.name: slice(None, None, -1)})

        if 'level' not in dataset.indexes:
            dataset = dataset.expand_dims(dim={"level": 1}, axis=1)
        else:
            if 't2m' in data_name:
            # If 'level' exists but is not needed, squeeze it to size 1 (drop all but one level)
                dataset = dataset.isel(level=0)
                dataset = dataset.expand_dims(dim={"level": 1}, axis=1)
            else:
                dataset = dataset.sel(level=np.array(levels))

        # Select correct variable key depending on storage
        if data_name in data_keys_map:
            var_key = data_keys_map[data_name]
        else:
            if isinstance(self.data_root, str) and self.data_root.endswith('.zarr'):
                var_key = data_map.get(data_name, data_name)
            else:
                var_key = data_name
        arr = dataset.get(var_key)
        if arr is None:
            # Fallback: try mapped name or common aliases
            fallback_keys = []
            if data_name in data_map:
                fallback_keys.append(data_map[data_name])
            # add a few common possibilities
            if data_name == 't2m':
                fallback_keys += ['t2m', '2m_temperature']
            for k in fallback_keys:
                tmp = dataset.get(k)
                if tmp is not None:
                    arr = tmp
                    break
        assert arr is not None, f"Variable '{var_key}' not found in dataset. Available: {list(dataset.data_vars.keys())}"
        data = arr.values

        mean = data.mean().reshape(1, 1, 1)
        std = data.std().reshape(1, 1, 1)
        # mean = dataset.mean('time').mean(('lat', 'lon')).compute()[data_name].values
        # std = dataset.std('time').mean(('lat', 'lon')).compute()[data_name].values
        data = (data - mean) / std

        return data, mean, std

    def _augment_seq(self, seqs, crop_scale=0.96):
        """Augmentations as a video sequence"""
        _, _, h, w = seqs.shape  # original shape, e.g., [4, 1, 128, 256]
        seqs = F.interpolate(seqs, scale_factor=1 / crop_scale, mode='bilinear')
        _, _, ih, iw = seqs.shape
        # Random Crop
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        seqs = seqs[:, :, x:x+h, y:y+w]
        # Random Flip
        if random.randint(0, 1):
            seqs = torch.flip(seqs, dims=(3, ))  # horizontal flip
        return seqs

    def __len__(self):
        return self.valid_idx.shape[0]

    def __getitem__(self, index):
        index = self.valid_idx[index]
        data = torch.tensor(self.data[index+self.idx_in])
        labels = torch.tensor(self.data[index+self.idx_out])
        if self.use_augment:
            len_data = self.idx_in.shape[0]
            seqs = self._augment_seq(torch.cat([data, labels], dim=0), crop_scale=0.96)
            data = seqs[:len_data, ...]
            labels = seqs[len_data:, ...]
        return data, labels


def load_data(batch_size,
              val_batch_size,
              data_root,
              num_workers=4,
              data_split='1_40625',
              data_name='t2m',
              train_time=None,
              val_time=None,
              test_time=None,
              idx_in=None,
              idx_out=None,
              step=1,
              levels=None,
              distributed=False, use_augment=False, use_prefetcher=False, drop_last=False,
              **kwargs):

    assert data_split in ['5_625', '2_8125', '1_40625', '0_25']
    # default to provided data_root; if a file is given, keep as-is; else try known dirs
    weather_dataroot = data_root
    if isinstance(data_root, str) and (data_root.endswith('.nc') or data_root.endswith('.zarr')):
        weather_dataroot = data_root
    else:
        for suffix in [f'weather_{data_split}deg', f'weather', f'{data_split}deg']:
            candidate = osp.join(data_root, suffix)
            if osp.exists(candidate):
                weather_dataroot = candidate

    # derive indices if not provided
    pre = kwargs.get('pre_seq_length', None)
    aft = kwargs.get('aft_seq_length', None)
    if idx_in is None and pre is not None:
        idx_in = [(-pre + 1 + i) * step for i in range(int(pre))]
    if idx_out is None and aft is not None:
        idx_out = [step * (i + 1) for i in range(int(aft))]

    train_set = WeatherBenchDataset(data_root=weather_dataroot,
                                    data_name=data_name, data_split=data_split,
                                    training_time=train_time,
                                    idx_in=idx_in,
                                    idx_out=idx_out,
                                    step=step, levels=levels, use_augment=use_augment)
    vali_set = WeatherBenchDataset(weather_dataroot,
                                    data_name=data_name, data_split=data_split,
                                    training_time=val_time,
                                    idx_in=idx_in,
                                    idx_out=idx_out,
                                    step=step, levels=levels, use_augment=False,
                                    mean=train_set.mean,
                                    std=train_set.std)
    test_set = WeatherBenchDataset(weather_dataroot,
                                    data_name, data_split=data_split,
                                    training_time=test_time,
                                    idx_in=idx_in,
                                    idx_out=idx_out,
                                    step=step, levels=levels, use_augment=False,
                                    mean=train_set.mean,
                                    std=train_set.std)

    dataloader_train = create_loader(train_set,
                                     batch_size=batch_size,
                                     shuffle=True, is_training=True,
                                     pin_memory=True, drop_last=drop_last,
                                     num_workers=num_workers,
                                     distributed=distributed, use_prefetcher=use_prefetcher)
    dataloader_vali = create_loader(vali_set,
                                    batch_size=val_batch_size,
                                    shuffle=False, is_training=False,
                                    pin_memory=True, drop_last=drop_last,
                                    num_workers=num_workers,
                                    distributed=distributed, use_prefetcher=use_prefetcher)
    dataloader_test = create_loader(test_set,
                                    batch_size=val_batch_size,
                                    shuffle=False, is_training=False,
                                    pin_memory=True, drop_last=drop_last,
                                    num_workers=num_workers,
                                    distributed=distributed, use_prefetcher=use_prefetcher)

    return dataloader_train, dataloader_vali, dataloader_test




def get_dataset(dataname, batch_size, val_batch_size, num_workers, data_root, dist=False, **kwargs):
    data_split_pool = ['5_625', '2_8125', '1_40625', '0_25']
    data_split = '5_625'
    for k in data_split_pool:
        if dataname.find(k) != -1:
            data_split = k
            # Convert 0p25 to 0_25 for internal processing
            if data_split == '0p25':
                data_split = '0_25'
    _kwargs = dict(kwargs)
    if 'data_split' in _kwargs:
        _kwargs.pop('data_split')
    if 'data_name' in _kwargs:
        _kwargs['data_name'] = _kwargs['data_name']
    if 'train_time' in _kwargs:
        _kwargs['train_time'] = _kwargs['train_time']
    if 'val_time' in _kwargs:
        _kwargs['val_time'] = _kwargs['val_time']
    if 'test_time' in _kwargs:
        _kwargs['test_time'] = _kwargs['test_time']
    if 'levels' in _kwargs:
        _kwargs['levels'] = _kwargs['levels']
    return load_data(batch_size, val_batch_size, data_root, num_workers,
                        distributed=dist, data_split=data_split, **_kwargs)






# if __name__ == '__main__':
#     from openstl.core import metric
#     data_split=['5_625', '1_40625']
#     data_name = 't2m'
#     # data_split=['5_625',]
#     # data_name = 'mv'

#     for _split in data_split:
#         step, levels = 24, [150, 500, 850]
#         dataloader_train, _, dataloader_test = \
#             load_data(batch_size=128,
#                     val_batch_size=32,
#                     data_root='../../data',
#                     num_workers=4, data_name=data_name,
#                     data_split=_split,
#                     train_time=['1979', '2015'],
#                     val_time=['2016', '2016'],
#                     test_time=['2017', '2018'],
#                     idx_in=[-11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0],
#                     idx_out=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
#                     step=step, levels=levels, use_augment=True)

#         print(len(dataloader_train), len(dataloader_test))
#         for item in dataloader_train:
#             print('train', item[0].shape)
#             if 'mv' in data_name:
#                 _, log = metric(item[0].cpu().numpy(), item[1].cpu().numpy(),
#                                 channel_names=dataloader_train.dataset.data_name, spatial_norm=True)
#                 print(log)
#             break
#         for item in dataloader_test:
#             print('test', item[0].shape)
#             break
