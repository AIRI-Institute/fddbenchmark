from fddbenchmark import FDDDataset

def test_small_tep():
    dataset = FDDDataset(name='small_tep')
    assert dataset.df.shape == (153300, 52)
    assert dataset.labels.shape == (153300,)