# Copyright (c) CAIRI AI Lab. All rights reserved

def load_data(dataname, batch_size, val_batch_size, num_workers, data_root, dist=False, **kwargs):
    cfg_dataloader = dict(
        pre_seq_length=kwargs.get('pre_seq_length', 10),
        aft_seq_length=kwargs.get('aft_seq_length', 10),
        in_shape=kwargs.get('in_shape', None),
        distributed=dist,
        use_augment=kwargs.get('use_augment', False),
        use_prefetcher=kwargs.get('use_prefetcher', False),
        drop_last=kwargs.get('drop_last', False),
    )

    if dataname == 'bair':
        from .dataloader_bair import load_data
        return load_data(batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader)
    elif dataname == 'human':
        from .dataloader_human import load_data
        return load_data(batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader)
    elif dataname == 'kitticaltech':
        from .dataloader_kitticaltech import load_data
        return load_data(batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader)
    elif 'kth' in dataname:  # 'kth', 'kth20', 'kth40'
        from .dataloader_kth import load_data
        return load_data(batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader)
    elif dataname in ['mmnist', 'mfmnist', 'mmnist_cifar']:  # 'mmnist', 'mfmnist', 'mmnist_cifar'
        from .dataloader_moving_mnist import load_data
        cfg_dataloader['data_name'] = kwargs.get('data_name', 'mnist')
        return load_data(batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader)
    elif 'noisymmnist' in dataname:  # 'mmnist - perceptual', 'mmnist - missing', 'mmnist - dynamic' 
        from .dataloader_noisy_moving_mnist import load_data
        cfg_dataloader['noise_type'] = kwargs.get('noise_type', 'perceptual')
        return load_data(batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader)
    elif 'kinetics' in dataname:  # 'kinetics400', 'kinetics600'
        from .dataloader_kinetics import load_data
        cfg_dataloader['data_name'] = kwargs.get('data_name', 'kinetics400')
        return load_data(batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader)
    elif dataname == 'taxibj':
        from .dataloader_taxibj import load_data
        return load_data(batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader)
    elif 'weather' in dataname:  # 'weather', 'weather_t2m', etc.
        from .dataloader_weather import load_data
        data_split_pool = ['5_625', '2_8125', '1_40625']
        data_split = '5_625'
        for k in data_split_pool:
            if dataname.find(k) != -1:
                data_split = k
        _kwargs = dict(kwargs)
        if 'data_split' in _kwargs:
            _kwargs.pop('data_split')
        # Derive idx_in/idx_out if not explicitly provided, using step & seq lengths
        pre = _kwargs.get('pre_seq_length', None)
        aft = _kwargs.get('aft_seq_length', None)
        step = int(_kwargs.get('time_interval', _kwargs.get('step', 1)))
        if pre is not None and aft is not None and (_kwargs.get('idx_in') is None or _kwargs.get('idx_out') is None):
            if _kwargs.get('idx_in') is None:
                _kwargs['idx_in'] = [(-pre + 1 + i) * step for i in range(pre)]
            if _kwargs.get('idx_out') is None:
                _kwargs['idx_out'] = [step * (i + 1) for i in range(aft)]
            # Ensure total_length consistency for downstream modules that may rely on it
            _kwargs['total_length'] = int(pre) + int(aft)
        # map parser-provided names to dataloader_weather names when present
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
    elif 'sevir' in dataname:  #'sevir_vis', 'sevir_ir069', 'sevir_ir107', 'sevir_vil'
        from .dataloader_sevir import load_data
        cfg_dataloader['data_name'] = kwargs.get('data_name', 'sevir')
        return load_data(batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader)
    else:
        raise ValueError(f'Dataname {dataname} is unsupported')