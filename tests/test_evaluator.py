from fddbenchmark import FDDDataset, FDDDataloader, FDDEvaluator
import numpy as np
import pandas as pd

def test_small_tep():
    dataset = FDDDataset(name='small_tep')
    loader = FDDDataloader(
        dataset.df,
        dataset.test_mask,
        dataset.labels,
        window_size=100,
        step_size=1,
        minibatch_training=True,
        batch_size=1024,
    )
    
    np.random.seed(0)
    pred = []
    label = []
    for ts, time_index, _label in loader:
        _pred = np.random.randint(21, size=time_index.shape[0])
        pred.append(pd.Series(_pred, index=time_index, dtype=int))
        label.append(_label)
    pred = pd.concat(pred)
    label = pd.concat(label)
    
    evaluator = FDDEvaluator(
        step_size=loader.step_size
    )

    metrics = evaluator.evaluate(label, pred)
    assert np.isclose(metrics['detection']['TPR'], 0.9535, atol=1e-4)
    assert np.isclose(metrics['detection']['FPR'], 0.9511, atol=1e-4)
    assert metrics['detection']['ADD'] == 0.05
    assert np.isclose(metrics['diagnosis']['CDR_total'], 0.0510, atol=1e-4)
    assert metrics['confusion_matrix'].shape == (21, 21)
    assert np.isclose(metrics['clustering']['ACC'], 0.0532, atol=1e-4)
    assert np.isclose(metrics['clustering']['NMI'], 0.0008, atol=1e-4)
    assert np.isclose(metrics['clustering']['ARI'], 0.0000007, atol=1e-7)
    assert metrics['classification']['TPR'].shape == (21,)
    assert metrics['classification']['FPR'].shape == (21,)
