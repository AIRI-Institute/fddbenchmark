import requests
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
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
            self.n_faults = 20
        else:
            raise Exception(
                f'{name} is an unknown dataset. Available datasets are: {available_datasets_str}'
            )
    def load_small_tep(self):
        if self.splitting_type == 'supervised':
            download_files(['dataset.csv', 'labels.csv', 'train_mask.csv', 'test_mask.csv'])
            self.df = pd.read_csv('dataset.csv', index_col=['run_id', 'sample'])
            self.labels = pd.read_csv('labels.csv', index_col=['run_id', 'sample'])['labels']
            self.train_mask = pd.read_csv('train_mask.csv', index_col=['run_id', 'sample'])['train_mask']
            self.test_mask = pd.read_csv('test_mask.csv', index_col=['run_id', 'sample'])['test_mask']
        if self.splitting_type == 'unsupervised':
            download_files(['dataset.csv', 'labels.csv', 'test_mask.csv', 'normal_train_mask.csv'])
            self.df = pd.read_csv('dataset.csv', index_col=['run_id', 'sample'])
            self.labels = pd.read_csv('labels.csv', index_col=['run_id', 'sample'])['labels']
            self.train_mask = pd.read_csv('normal_train_mask.csv', index_col=['run_id', 'sample'])['train_mask']
            self.test_mask = pd.read_csv('test_mask.csv', index_col=['run_id', 'sample'])['test_mask']
        if self.splitting_type == 'semisupervised':
            download_files([
                'dataset.csv', 
                'labels.csv', 
                'test_mask.csv', 
                'train_mask.csv', 
                'unlabeled_train_mask.csv'
            ])
            self.df = pd.read_csv('dataset.csv', index_col=['run_id', 'sample'])
            self.labels = pd.read_csv('labels.csv', index_col=['run_id', 'sample'])['labels']
            self.train_mask = pd.read_csv('train_mask.csv', index_col=['run_id', 'sample'])['train_mask']
            self.test_mask = pd.read_csv('test_mask.csv', index_col=['run_id', 'sample'])['test_mask']
            unlabeled_train_mask = pd.read_csv('unlabeled_train_mask.csv', index_col=['run_id', 'sample'])\
                ['unlabeled_train_mask']
            self.labels.loc[unlabeled_train_mask] = np.nan

def download_files(file_names):
    for fname in tqdm(file_names, desc='Downloading'):
        url = f'https://raw.githubusercontent.com/airi-industrial-ai/fdd-datasets/main/small_tep/{fname}'
        open(fname, 'wb').write(requests.get(url).content)

class FDDDataloader():
    def __init__(
        self, 
        dataframe: pd.DataFrame, 
        labels: pd.Series,
        window_size: int, 
        step_size: int, 
        minibatch_training=False, 
        batch_size=0, 
        shuffle=False,
    ):
        assert batch_size if minibatch_training else True
        self.df = dataframe
        self.labels = labels
        assert np.all(self.labels.index == self.df.index)
        self.window_size = window_size
        self.step_size = step_size
        assert self.step_size <= self.window_size
        sample_seq = []
        for run_id in self.df.index.get_level_values(0).unique():
            _idx = self.df.index.get_locs([run_id])
            sample_seq.extend(
                np.arange(_idx.min(), _idx.max() - self.window_size + 1, self.step_size)
            )
        self.sample_seq = np.random.permutation(sample_seq) if shuffle else np.array(sample_seq)
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
            # preparing batch of labels
            sample_ids = self.sample_seq[self.batch_seq[self.iter]:self.batch_seq[self.iter+1]]
            row_idx = np.tile(sample_ids[:, None], (1, self.window_size)) + np.arange(self.window_size)
            row_isna = np.isnan(self.labels.values[row_idx]).min(axis=1)
            labels_batch = np.zeros(row_isna.shape[0])
            labels_batch[row_isna] = np.nan
            if ~row_isna.any():
                # maximum label reduction: if at least a single time stamp is fault
                # then the entire sample is fault
                labels_batch[~row_isna] = self.labels.values[row_idx][~row_isna].max(axis=1)
            # an index of a sample is an index of the last time stamp in the sample
            index_batch = self.labels.index[row_idx.max(axis=1)]
            labels_batch = pd.Series(labels_batch, name='labels', index=index_batch)
            # preparing batch of time series
            row_idx = np.tile(row_idx[..., None], (1, 1, self.df.shape[1]))
            col_idx = np.arange(self.df.shape[1])[None, None, :]
            col_idx = np.tile(col_idx, (row_idx.shape[0], self.window_size, 1))
            ts_batch = self.df.values[row_idx, col_idx]
            self.iter += 1
            return ts_batch, labels_batch.index, labels_batch
        else:
            raise StopIteration

class FDDEvaluator():
    def __init__(self, window_size: int, step_size: int):
        self.window_size = window_size
        self.step_size = step_size
        
    def evaluate(self, labels, pred):
        # labels should be non-negative integer values, normal is 0
        assert np.all(np.sort(np.unique(labels)) == np.arange(labels.max() + 1))
        fdd_cm = confusion_matrix(labels, pred, labels=np.arange(labels.max() + 1))
        metrics = {'detection': dict(), 'diagnosis': dict()}
        tp = fdd_cm[1:, 1:].sum()
        total = fdd_cm.sum()
        metrics['confusion_matrix'] = fdd_cm
        metrics['detection']['TPR'] = tp / total
        metrics['detection']['FPR'] = fdd_cm[0, 1:].sum() / total

        pred_change_point = pred[pred != 0]
        pred_change_point = pred_change_point.reset_index()\
            .groupby(['run_id'])['sample'].min() + self.window_size
        real_change_point = labels[labels != 0]
        real_change_point = real_change_point.reset_index()\
            .groupby(['run_id'])['sample'].min() + self.window_size - self.step_size
        detection_delay = (pred_change_point - real_change_point)
        valid_delay = detection_delay[detection_delay > 0]

        metrics['detection']['ADD'] = valid_delay.mean()
        metrics['detection']['VDR'] = valid_delay.shape[0] / real_change_point.shape[0]

        correct_diagnoses = fdd_cm[1:, 1:].diagonal()
        metrics['diagnosis']['CDR'] = correct_diagnoses / tp
        metrics['diagnosis']['CDR_total'] = correct_diagnoses.sum() / tp
        metrics['diagnosis']['MDR'] = (tp - correct_diagnoses.sum()) / tp
        return metrics
    
    def print_metrics(self, labels, pred):
        metrics = self.evaluate(labels, pred)
        print('Detection metrics \n----------')
        print('True Positive Rate (TPR): {:.4f}'.format(metrics['detection']['TPR']))
        print('False Positive Rate (FPR): {:.4f}'.format(metrics['detection']['FPR']))
        print('Average Detection Delay (ADD): {:.2f}'.format(metrics['detection']['ADD']))
        print('Valid Delay Rate (VDR): {:.4f}'.format(metrics['detection']['VDR']))
        print('\nDiagnosis metrics \n----------')
        print('Correct Diagnosis Rate (CDR):')
        for i in np.arange(labels.max()).astype('int'):
            print('    Fault {:02d}: {:.4f}'.format(i+1, metrics['diagnosis']['CDR'][i]))
        print('Total Correct Diagnosis Rate (Total CDR): {:.4f}'.format(metrics['diagnosis']['CDR_total']))
        print('Misdiagnosis Rate (MDR): {:.4f}'.format(metrics['diagnosis']['MDR']))