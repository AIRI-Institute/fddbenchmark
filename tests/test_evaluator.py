from fddbenchmark import FDDDataset, FDDDataloader, FDDEvaluator
import numpy as np
import pandas as pd


def test_small_tep():
    dataset = FDDDataset(name='small_tep')
    loader = FDDDataloader(
        dataset.df,
        dataset.test_mask,
        dataset.label,
        window_size=100,
        step_size=1,
        use_minibatches=True,
        batch_size=1024,
    )
    
    np.random.seed(0)
    pred = []
    label = []
    for _, time_index, _label in loader:
        _pred = np.random.randint(21, size=time_index.shape[0])
        pred.append(pd.Series(_pred, index=time_index, dtype=int))
        label.append(pd.Series(_label, index=time_index, dtype=int))
    pred = pd.concat(pred)
    label = pd.concat(label)

    evaluator = FDDEvaluator(
        step_size=loader.step_size
    )

    metrics = evaluator.evaluate(label, pred)
    assert np.isclose(metrics['detection']['TPR'], 0.9535, atol=1e-4)
    assert np.isclose(metrics['detection']['FPR'], 0.9517, atol=1e-4)
    assert metrics['detection']['ADD'] == 0.03
    assert np.isclose(metrics['diagnosis']['CDR_total'], 0.0514, atol=1e-4)
    assert metrics['confusion_matrix'].shape == (21, 21)
    assert np.isclose(metrics['clustering']['ACC'], 0.0531, atol=1e-4)
    assert np.isclose(metrics['clustering']['NMI'], 0.0008, atol=1e-4)
    assert np.isclose(metrics['clustering']['ARI'], -18e-7, atol=1e-7)
    assert metrics['classification']['TPR'].shape == (21,)
    assert metrics['classification']['FPR'].shape == (21,)
