from fddbenchmark import FDDDataset, FDDDataloader
import numpy as np

def test_small_tep():
    dataset = FDDDataset(name='small_tep')
    loader = FDDDataloader(
        dataset.df,
        dataset.train_mask,
        dataset.labels,
        window_size=100,
        step_size=1,
        minibatch_training=True,
        batch_size=1024,
    )
    assert len(loader) == 42
    for ts, time_index, label in loader:
        break
    assert ts.shape == (1024, 100, 52)
    assert np.all(time_index == label.index)
    assert np.all(ts[0, 0, :5] == [2.5038e-01, 3.6740e+03, 4.5290e+03, 9.2320e+00, 2.6889e+01])