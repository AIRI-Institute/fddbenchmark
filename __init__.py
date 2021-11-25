import requests
import pandas as pd
import numpy as np
from tqdm import tqdm

class FDDDataset():
    def __init__(self, name, splitting_type):
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
            download_files(['dataset.csv', 'labels.csv', 'test_mask.txt', 'train_mask.txt', 'unlabeled_train_mask.txt'])
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