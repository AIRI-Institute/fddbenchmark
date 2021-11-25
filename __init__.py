import requests
import pandas as pd
import numpy as np
from tqdm import tqdm

class FDDDataset():
    def __init__(self, name, splitting_type='supervised'):
        self.splitting_type = splitting_type
        self.df = None
        self.labels = None
        self.train_mask = None
        self.test_mask = None
        available_datasets = ['small_tep']
        available_datasets_str = ', '.join(available_datasets)
        if name == 'small_tep':
            self.load_small_tep()
        else:
            raise Exception(f'{name} is an unknown dataset. Available datasets are: {available_datasets_str}')
    def load_small_tep(self):
        if self.splitting_type == 'supervised':
            file_names = ['dataset.csv', 'labels.txt', 'train_mask.txt', 'test_mask.txt']
            for fname in tqdm(file_names):
                url = f'https://raw.githubusercontent.com/airi-industrial-ai/fdd-datasets/main/small_tep/{fname}'
                open(fname, 'wb').write(requests.get(url).content)
            self.df = pd.read_csv('dataset.csv')
            self.labels = np.loadtxt('labels.txt', dtype=int)
            self.train_mask = np.loadtxt('train_mask.txt', dtype=bool)
            self.test_mask = np.loadtxt('test_mask.txt', dtype=bool)
    def _repr_html_(self):
        return self.df.head()._repr_html_()