import requests
import pandas as pd
import numpy as np
from tqdm import tqdm

class FDDDataset():
    def __init__(self, name: str, splitting_type: str):
        self.name = name
        self.splitting_type = splitting_type
        self.df = None
        self.labels = None
        self.train_mask = None
        self.test_mask = None
        available_datasets = ['small_tep']
        available_datasets_str = ', '.join(available_datasets)
        if self.name == 'small_tep':
            self.load_small_tep()
        else:
            raise Exception(
                f'{name} is an unknown dataset. Available datasets are: {available_datasets_str}'
            )
    def load_small_tep(self):
        if self.splitting_type == 'supervised':
            download_files(['dataset.csv', 'labels.csv', 'train_mask.txt', 'test_mask.txt'])
            self.df = pd.read_csv('dataset.csv')
            self.labels = pd.read_csv('labels.csv')
            self.train_mask = np.loadtxt('train_mask.txt', dtype=bool)
            self.test_mask = np.loadtxt('test_mask.txt', dtype=bool)
        if self.splitting_type == 'unsupervised':
            download_files(['dataset.csv', 'labels.csv', 'test_mask.txt', 'normal_train_mask.txt'])
            self.df = pd.read_csv('dataset.csv')
            self.labels = pd.read_csv('labels.csv')
            self.train_mask = np.loadtxt('normal_train_mask.txt', dtype=bool)
            self.test_mask = np.loadtxt('test_mask.txt', dtype=bool)
        if self.splitting_type == 'semisupervised':
            download_files([
                'dataset.csv', 
                'labels.csv', 
                'test_mask.txt', 
                'train_mask.txt', 
                'unlabeled_train_mask.txt'
            ])
            self.df = pd.read_csv('dataset.csv')
            self.labels = pd.read_csv('labels.csv')
            self.train_mask = np.loadtxt('train_mask.txt', dtype=bool)
            self.test_mask = np.loadtxt('test_mask.txt', dtype=bool)
            unlabeled_train_mask = np.loadtxt('unlabeled_train_mask.txt', dtype=bool)
            self.labels.loc[unlabeled_train_mask, ['fault_type']] = None
    def _repr_html_(self):
        return self.df.head()._repr_html_()

def download_files(file_names):
    for fname in tqdm(file_names, desc='Downloading'):
        url = f'https://raw.githubusercontent.com/airi-industrial-ai/fdd-datasets/main/small_tep/{fname}'
        open(fname, 'wb').write(requests.get(url).content)

class FDDDataloader():
    def __init__(
        self, 
        dataset: FDDDataset, 
        window_size: int, 
        step_size: int, 
        minibatch_training: bool, 
        batch_size: int, 
        shuffle: bool,
    ):
        self.ds = dataset
        self.window_size = 10
        assert step_size <= window_size
        sample_seq = np.arange(0, self.ds.df.shape[0] - self.window_size + 1, step_size)
        self.sample_seq = np.random.permutation(sample_seq) if shuffle else sample_seq
        n_samples = len(sample_seq)
        batch_seq = list(range(0, n_samples, batch_size)) if minibatch_training else [0]
        if batch_seq[-1] < n_samples: 
            batch_seq.append(n_samples)
        self.n_batches = len(batch_seq) - 1
        self.batch_seq = np.array(batch_seq)
    
    def __len__(self):
        return self.n_batches
    
    def __iter__(self):
        self.iter = 0
        return self

    def __next__(self):
        if self.iter < self.n_batches:
            sample_ids = self.sample_seq[self.batch_seq[self.iter]:self.batch_seq[self.iter+1]]
            row_idx = np.tile(sample_ids[:, None], (1, self.window_size)) + np.arange(self.window_size)
            row_idx = np.tile(row_idx[..., None], (1, 1, self.ds.df.shape[1]))
            col_idx = np.arange(self.ds.df.shape[1])[None, None, :]
            col_idx = np.tile(col_idx, (row_idx.shape[0], self.window_size, 1))
            batch = self.ds.df.values[row_idx, col_idx]
            self.iter += 1
            return batch
        else:
            raise StopIteration
