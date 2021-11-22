import requests
import pandas as pd

class FDDDataset(pd.DataFrame):
    def __init__(self, name):
        available_datasets = ['small_tep_train', 'small_tep_test']
        available_datasets_str = ', '.join(available_datasets)
        if name == 'small_tep_train':
            url = 'https://raw.githubusercontent.com/airi-industrial-ai/fdd-datasets/main/small_tep/training.csv'
            open('training.csv', 'wb').write(requests.get(url).content)
            df = pd.read_csv('training.csv')
        elif name == 'small_tep_test':
            url = 'https://raw.githubusercontent.com/airi-industrial-ai/fdd-datasets/main/small_tep/testing.csv'
            open('testing.csv', 'wb').write(requests.get(url).content)
            df = pd.read_csv('testing.csv')
        else:
            raise Exception(f'{name} is an unknown dataset. Available datasets are: {available_datasets_str}')
        super().__init__(data=df)