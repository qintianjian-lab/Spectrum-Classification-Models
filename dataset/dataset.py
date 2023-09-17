import os

import numpy as np
import pandas as pd
import torch
from sklearn.utils import shuffle


class SpectrumDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_dir: str,
                 spectrum_dir: str,
                 load_mode: str,
                 label_dir: str,
                 type_list: list,
                 spectrum_size: int,
                 ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.spectrum_dir = spectrum_dir
        self.load_mode = load_mode
        assert self.load_mode in ['train', 'val', 'test'], '[Error] load_mode must be in [train, val, test]'
        self.label_dir = shuffle(pd.read_csv(os.path.join(dataset_dir, load_mode, label_dir, 'label.csv'), header=0))
        self.type_list = type_list
        self.spectrum_size = spectrum_size

    def __len__(self):
        return len(self.label_dir)

    def __getitem__(self, idx):
        _row = self.label_dir.iloc[idx]
        data_id, data_label = _row['basename'], _row['label']
        assert data_label in self.type_list, '[Error] data_label must be in {}'.format(self.type_list)
        data_label = self.type_list.index(data_label)
        # load spectrum
        spectrum_path = os.path.join(self.dataset_dir, self.load_mode, self.spectrum_dir, data_id + '.csv')
        spectrum = np.loadtxt(spectrum_path, delimiter=',', dtype=np.float32)[:, 1]
        assert spectrum.shape[0] <= self.spectrum_size, '[Error] spectrum_size must be larger than spectrum.shape[0]' \
                                                        ''
        # padding
        padding_half = (self.spectrum_size - spectrum.shape[0]) // 2
        spectrum = torch.cat(
            (torch.zeros(padding_half),
             torch.from_numpy(spectrum).float(),
             torch.zeros(padding_half))).unsqueeze(0)
        # z-score
        # spectrum = (spectrum - spectrum.mean()) / spectrum.std()
        # spectrum = (spectrum - spectrum.min()) / (spectrum.max() - spectrum.min())
        return spectrum, data_label, data_id


def build_dataloader(config: dict, mode: str, cross_val_name: str = '') -> torch.utils.data.DataLoader:
    """
    build dataloader
    :param config: contains keys: dataset_dir, spectrum_dir, label_dir, type_list, spectrum_size, batch_size, num_workers
    :param mode: dataset load mode, must be in ['train', 'val', 'test']
    :param cross_val_name: cross validation name
    :return: dataloader
    """
    assert mode in ['train', 'val', 'test'], '[Error] mode must be "train", "val" or "test"'
    # check config
    keys = [
        # dataset
        'dataset_dir',
        'spectrum_dir',
        'label_dir',
        'type_list',
        'spectrum_size',
        # dataloader
        'batch_size',
        'num_workers',
    ]
    for key in keys:
        assert key in config, f'[Error] {key} must be in config'
    if mode == 'test':
        print('[Info] mode is test, "batch_size" will be set to 1')
        config['batch_size'] = 1
    dataset_dir = config['dataset_dir']
    if cross_val_name is not None and cross_val_name != '':
        dataset_dir = os.path.join(config['dataset_dir'], cross_val_name)
    _dataset = SpectrumDataset(
        dataset_dir=dataset_dir,
        spectrum_dir=config['spectrum_dir'],
        load_mode=mode,
        label_dir=config['label_dir'],
        type_list=config['type_list'],
        spectrum_size=config['spectrum_size'],
    )
    _dataloader = torch.utils.data.DataLoader(
        dataset=_dataset,
        batch_size=config['batch_size'],
        shuffle=mode == 'train',
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=False
    )
    return _dataloader
