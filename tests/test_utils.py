from deepinterpolation import utils
import numpy as np

def test_dump_as_dat2():
    file_name = 'test.dat2'
    X = 10*np.random.randn(3,4)
    X = np.int16(X)
    utils.dump_as_dat2(file_name, X)
    
def test_load_dat2():
    test_dump_as_dat2()
    fp = utils.load_dat2('test.dat2', np.int16, (3, 4))
    assert fp.shape == (3, 4), 'Invalid data shape'
    
def test_generate_dataset():
    X, N = utils.generate_dataset(10, (5, 5), 1e3, 1, waveform='sine')
    assert X.shape == (10*1e3, 5, 5), 'Invalid data shape'
    assert N.shape == (10*1e3, 5, 5), 'Invalid data shape'
    
    
    