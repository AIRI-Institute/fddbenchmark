import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment as linear_assignment
from tqdm import tqdm
import zipfile
import os
import requests

class FDDDataset():
    def __init__(self, name: str, splitting_type: str):
        self.name = name
        self.splitting_type = splitting_type
        self.df = None
        self.labels = None
        self.train_mask = None
        self.test_mask = None
        available_datasets = ['small_tep', 'reinartz_tep', 'rieth_tep']
        available_datasets_str = ', '.join(available_datasets)
        if self.name not in available_datasets:
            raise Exception(
                f'{name} is an unknown dataset. Available datasets are: {available_datasets_str}'
            )
        if self.name == 'small_tep':
            self.load_small_tep()
        if self.name == 'reinartz_tep':
            self.load_reinartz_tep()
        if self.name == 'rieth_tep':
            self.load_rieth_tep()
        
    def load_small_tep(self):
        ref_path = 'data/small_tep/'
        if not os.path.exists(ref_path):
            os.makedirs(ref_path)
        url = "https://industrial-makarov.obs.ru-moscow-1.hc.sbercloud.ru/small_tep.zip"
        zfile_path = 'data/small_tep.zip'
        if not os.path.exists(zfile_path):
            download_pgbar(url, zfile_path, fname='small_tep.zip')
        
        extracting_files(zfile_path, ref_path)
        self.df = read_csv_pgbar(ref_path + 'dataset.csv', index_col=['run_id', 'sample'])
        self.labels = read_csv_pgbar(ref_path + 'labels.csv', index_col=['run_id', 'sample'])['labels']
        train_mask = read_csv_pgbar(ref_path + 'train_mask.csv', index_col=['run_id', 'sample'])['train_mask']
        test_mask = read_csv_pgbar(ref_path + 'test_mask.csv', index_col=['run_id', 'sample'])['test_mask']
        self.train_mask = train_mask.astype('boolean')
        self.test_mask = test_mask.astype('boolean')
        
        if self.splitting_type == 'supervised':
            pass
        if self.splitting_type == 'unsupervised':
            self.labels[self.train_mask] = np.nan
        if self.splitting_type == 'semisupervised':
            unlabeled_train_mask = read_csv_pgbar(
                ref_path + 'unlabeled_train_mask.csv', 
                index_col=['run_id', 'sample'])['unlabeled_train_mask']
            self.labels.loc[unlabeled_train_mask.astype('boolean')] = np.nan
    
    def load_reinartz_tep(self):
        ref_path = 'data/reinartz_tep/'
        if not os.path.exists(ref_path):
            os.makedirs(ref_path)
        url = "https://industrial-makarov.obs.ru-moscow-1.hc.sbercloud.ru/reinartz_tep.zip"
        zfile_path = 'data/reinartz_tep.zip'
        if not os.path.exists(zfile_path):
            download_pgbar(url, zfile_path, fname='reinartz_tep.zip')
        
        extracting_files(zfile_path, ref_path)
        self.df = read_csv_pgbar(ref_path + 'dataset.csv', index_col=['run_id', 'sample'])
        self.labels = read_csv_pgbar(ref_path + 'labels.csv', index_col=['run_id', 'sample'])['labels']
        train_mask = read_csv_pgbar(ref_path + 'train_mask.csv', index_col=['run_id', 'sample'])['train_mask']
        test_mask = read_csv_pgbar(ref_path + 'test_mask.csv', index_col=['run_id', 'sample'])['test_mask']
        self.train_mask = train_mask.astype('boolean')
        self.test_mask = test_mask.astype('boolean')
        
        if self.splitting_type == 'supervised':
            pass
        if self.splitting_type == 'unsupervised':
            self.labels[self.train_mask] = np.nan
        if self.splitting_type == 'semisupervised':
            labeled_train_mask = read_csv_pgbar(
                ref_path + 'labeled_train_mask.csv', 
                index_col=['run_id', 'sample'])['labeled_train_mask']
            self.labels.loc[~labeled_train_mask.astype('boolean')] = np.nan

    def load_rieth_tep(self):
        ref_path = 'data/rieth_tep/'
        if not os.path.exists(ref_path):
            os.makedirs(ref_path)
        url = "https://industrial-makarov.obs.ru-moscow-1.hc.sbercloud.ru/rieth_tep.zip"
        zfile_path = 'data/rieth_tep.zip'
        if not os.path.exists(zfile_path):
            download_pgbar(url, zfile_path, fname='rieth_tep.zip')
        
        extracting_files(zfile_path, ref_path)
        self.df = read_csv_pgbar(ref_path + 'dataset.csv', index_col=['run_id', 'sample'])
        self.labels = read_csv_pgbar(ref_path + 'labels.csv', index_col=['run_id', 'sample'])['labels']
        train_mask = read_csv_pgbar(ref_path + 'train_mask.csv', index_col=['run_id', 'sample'])['train_mask']
        test_mask = read_csv_pgbar(ref_path + 'test_mask.csv', index_col=['run_id', 'sample'])['test_mask']
        self.train_mask = train_mask.astype('boolean')
        self.test_mask = test_mask.astype('boolean')
        
        if self.splitting_type == 'supervised':
            pass
        if self.splitting_type == 'unsupervised':
            self.labels[self.train_mask] = np.nan
        if self.splitting_type == 'semisupervised':
            labeled_train_mask = read_csv_pgbar(
                ref_path + 'labeled_train_mask.csv', 
                index_col=['run_id', 'sample'])['labeled_train_mask']
            self.labels.loc[~labeled_train_mask.astype('boolean')] = np.nan

def extracting_files(zfile_path, ref_path):
    with zipfile.ZipFile(zfile_path, 'r') as zfile:
        for entry_info in zfile.infolist():
            if os.path.exists(ref_path + entry_info.filename):
                continue
            input_file = zfile.open(entry_info.filename)
            target_file = open(ref_path + entry_info.filename, 'wb')
            bsize = 1024 * 10000
            block = input_file.read(bsize)
            with tqdm(
                total=entry_info.file_size, 
                desc=f'Extracting {entry_info.filename}', 
                unit='B', 
                unit_scale=True, 
                unit_divisor=1024) as pbar:
                while block:
                    target_file.write(block)
                    block = input_file.read(bsize)
                    pbar.update(bsize)
            input_file.close()
            target_file.close()

def download_pgbar(url, zfile_path, fname):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("Content-Length"))
    with open(zfile_path, 'wb') as file: 
        with tqdm(
            total=total,
            desc=f'Downloading {fname}',
            unit='B',
            unit_scale=True,
            unit_divisor=1024) as pbar:
            for data in resp.iter_content(chunk_size=1024):
                file.write(data)
                pbar.update(len(data))

def read_csv_pgbar(csv_path, index_col, chunksize=1024*100):
    rows = sum(1 for _ in open(csv_path, 'r')) - 1
    chunk_list = []
    with tqdm(total=rows, desc=f'Reading {csv_path}') as pbar:
        for chunk in pd.read_csv(csv_path, index_col=index_col, chunksize=chunksize):
            chunk_list.append(chunk)
            pbar.update(len(chunk))
    df = pd.concat((f for f in chunk_list), axis=0)
    return df

class FDDDataloader():
    def __init__(
        self, 
        dataframe: pd.DataFrame, 
        mask: pd.Series, 
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
        for run_id in tqdm(
            self.labels[mask].index.get_level_values(0).unique(), 
            desc='Creating sequence of samples'):
            _idx = self.labels[mask].index.get_locs([run_id])
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
                # maximum label reduction: if at least a single value is fault
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
    def __init__(self, splitting_type: str, step_size: int):
        self.splitting_type = splitting_type
        self.step_size = step_size
        
    def evaluate(self, labels, pred):
        # labels should be non-negative integer values, normal is 0
        assert np.all(np.sort(np.unique(labels)) == np.arange(labels.max() + 1))
        # confustion matrix: rows are truth classes, columns are predicted classes
        fdd_cm = confusion_matrix(labels, pred, labels=np.arange(labels.max() + 1))
        metrics = {'detection': dict(), 'diagnosis': dict(), 'clustering': dict(), 'classification': dict()}
        metrics['confusion_matrix'] = fdd_cm
        metrics['detection']['TPR'] = fdd_cm[1:, 1:].sum() / fdd_cm[1:, :].sum()
        metrics['detection']['FPR'] = fdd_cm[0, 1:].sum() / fdd_cm[0, :].sum()

        real_change_point = labels[labels != 0].reset_index().groupby(['run_id'])['sample'].min()
        pred_change_point = pred[pred != 0].reset_index().set_index(['run_id'])['sample']
        pred_real_change_point = pd.merge(
            pred_change_point, 
            real_change_point, 
            how='left', 
            on='run_id', 
            suffixes=('', '_real')
        )
        valid_change_point = pred_real_change_point['sample_real'] <= pred_real_change_point['sample']
        pred_change_point = pred_change_point[valid_change_point].groupby(['run_id']).min()
        detection_delay = (pred_change_point - real_change_point) * self.step_size
        metrics['detection']['ADD'] = detection_delay.mean()
        
        correct_diagnosis = fdd_cm[1:, 1:].diagonal()
        tp = fdd_cm[1:, 1:].sum()
        metrics['diagnosis']['CDR'] = correct_diagnosis / fdd_cm[1:, 1:].sum(axis=1)
        metrics['diagnosis']['CDR_total'] = correct_diagnosis.sum() / tp
        metrics['diagnosis']['MDR'] = (tp - correct_diagnosis.sum()) / tp
        
        correct_classification = fdd_cm.diagonal()
        metrics['classification']['TPR'] = correct_classification / fdd_cm.sum(axis=1)
        metrics['classification']['FPR'] = fdd_cm[0] / fdd_cm[0].sum()
        
        metrics['clustering']['ACC'] = cluster_acc(labels.values, pred.values)
        metrics['clustering']['NMI'] = normalized_mutual_info_score(labels.values, pred.values)
        metrics['clustering']['ARI'] = adjusted_rand_score(labels.values, pred.values)
        return metrics
    
    def print_metrics(self, labels, pred):
        metrics = self.evaluate(labels, pred)
        print('Detection metrics\n-----------------')
        print('True Positive Rate (TPR): {:.4f}'.format(metrics['detection']['TPR']))
        print('False Positive Rate (FPR): {:.4f}'.format(metrics['detection']['FPR']))
        print('Average Detection Delay (ADD): {:.2f}'.format(metrics['detection']['ADD']))
        
        print('\nDiagnosis metrics\n-----------------')
        print('Correct Diagnosis Rate (CDR):')
        for i in np.arange(labels.max()).astype('int'):
            print('    Fault {:02d}: {:.4f}'.format(i+1, metrics['diagnosis']['CDR'][i]))
        print('Total Correct Diagnosis Rate (Total CDR): {:.4f}'.format(metrics['diagnosis']['CDR_total']))
        print('Misdiagnosis Rate (MDR): {:.4f}'.format(metrics['diagnosis']['MDR']))
        
        print('\nClassification metrics\n-----------------')
        print('TPR/FPR:')
        for i in np.arange(labels.max()).astype('int'):
            print('    Fault {:02d}: {:.4f}/{:.4f}'.format(i+1, metrics['classification']['TPR'][i+1], metrics['classification']['FPR'][i+1]))

        print('\nClustering metrics\n-----------------')
        print('Adjusted Rand Index (ARI): {:.4f}'.format(metrics['clustering']['ARI']))
        print('Normalized Mutual Information (NMI): {:.4f}'.format(metrics['clustering']['NMI']))
        print('Unsupervised Clustering Accuracy (ACC): {:.4f}'.format(metrics['clustering']['ACC']))

def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed. 
    Taken from https://github.com/XifengGuo/DEC-DA
    # Arguments
        y: true labels, numpy.array with shape (n_samples,)
        y_pred: predicted labels, numpy.array with shape (n_samples,)
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    
    return sum([w[ind[0][i], ind[1][i]] for i in range(ind[0].size)]) * 1.0 / y_pred.size