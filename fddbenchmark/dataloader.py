from typing import Optional
from tqdm import tqdm
import pandas as pd
import numpy as np


class FDDDataloader:
    def __init__(
            self,
            dataframe: pd.DataFrame,
            mask: pd.Series,
            label: pd.Series,
            window_size: int,
            dilation: int = 1,
            step_size: int = 1,
            use_minibatches: bool = False,
            batch_size: Optional[int] = None,
            shuffle: bool = False,
            random_state: Optional[int] = None,
    ) -> None:
        if dataframe.index.names != ['run_id', 'sample']:
            raise ValueError("``dataframe`` must have multi-index ('run_id', 'sample')")

        if not np.all(dataframe.index == mask.index) or not np.all(dataframe.index == label.index):
            raise ValueError("``dataframe`` and ``label`` must have the same indices.")

        if step_size > window_size:
            raise ValueError("``step_size`` must be less or equal to ``window_size``.")

        if use_minibatches and batch_size is None:
            raise ValueError("If you set ``use_minibatches=True``, "
                             "you must set ``batch_size`` to a positive number.")

        self.df = dataframe
        self.label = label
        self.window_size = window_size
        self.dilation = dilation
        self.step_size = step_size

        window_end_indices = []
        run_ids = dataframe[mask].index.get_level_values(0).unique()
        for run_id in tqdm(run_ids, desc='Creating sequence of samples'):
            indices = np.array(dataframe.index.get_locs([run_id]))
            indices = indices[self.window_size - 1:]
            indices = indices[::step_size]
            indices = indices[mask.iloc[indices].to_numpy(dtype=bool)]
            window_end_indices.extend(indices)

        if random_state is not None:
            np.random.seed(random_state)

        self.window_end_indices = np.random.permutation(window_end_indices) if shuffle else np.array(window_end_indices)

        n_samples = len(window_end_indices)
        batch_seq = list(range(0, n_samples, batch_size)) if use_minibatches else [0]
        batch_seq.append(n_samples)
        self.batch_seq = np.array(batch_seq)
        self.n_batches = len(batch_seq) - 1

    def __len__(self):
        return self.n_batches
    
    def __iter__(self):
        self.iter = 0
        return self

    def __next__(self):
        if self.iter < self.n_batches:
            ts_batch, index_batch, label_batch = self.__getitem__(self.iter)
            self.iter += 1
            return ts_batch, index_batch, label_batch
        else:
            raise StopIteration

    def __getitem__(self, idx):
        ends_indices = self.window_end_indices[self.batch_seq[idx]:self.batch_seq[idx + 1]]
        windows_indices = ends_indices[:, None] - np.arange(0, self.window_size, self.dilation)[::-1]

        ts_batch = self.df.values[windows_indices]  # (batch_size, window_size, ts_dim)
        label_batch = self.label.values[ends_indices]
        index_batch = self.label.index[ends_indices]

        return ts_batch, index_batch, label_batch
