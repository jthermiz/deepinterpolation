import numpy as np
import h5py
import matplotlib.pyplot as plt

def dump_as_dat2(file_name, X):
    """Dump long form data matrix as a dat2

    Parameters
    ----------
    file_name: str
        Path and file name where data will be dumped
    X : ndarray
        Data matrix, samples by channels
    """
    fp = np.memmap(file_name, dtype=X.dtype, mode='w+', shape=X.shape)
    fp[:] = X[:]
    fp.flush()
    
def load_dat2(file_name, dtype, shape):
    """Read dat2 from disk

    Parameters
    ----------
    file_name : str
        Path and file name where data is stored
    dtype : np.dtype
        Data type for numbers (eg. np.int16)
    shape : tuple
        Shape of data matrix

    Returns
    -------
    memmap
        Memory map to array stored on disk that is read-only
    """
    fp = np.memmap(file_name, dtype=dtype, mode='r', shape=shape)
    return fp

def generate_dataset(duration, shape, rate, snr, decay=1, waveform='sine', dtype=np.int16):
    """Generate dummy dataset for testing purposes

    Parameters
    ----------
    duration : numeric
        Duration in seconds
    shape : tuple
        Shape of synthetic electrode array
    rate : numeric
        Sample rate
    snr : numeric
        Signal to noise ratio
    decay : numeric
        Signal decay over space (larger numbers --> more decay), by default 1
    waveform : str, optional
        Waveform type, by default 'sine'    
    dtype : np.dtype
        Number datatype, by default 'np.int16'

    Returns
    -------
    ndarray
        Data cube of synthetic recordings

    Raises
    ------
    NotImplementedError
        For waveforms not implemented
    """
    scale_factor = 10e3
    frames = int(duration*rate)
    rows, cols = shape
    data_shape = [frames, rows, cols]
    X = np.zeros(data_shape)
    N = scale_factor/snr*np.random.randn(*data_shape)
    t = np.linspace(0, duration, frames)
    
    if waveform == 'sine':
        s = scale_factor*np.sin(2*np.pi*10*t)
        rctr, cctr = rows//2, cols//2
    else:
        raise NotImplementedError('That waveform has not been implemented')
        
    for r in range(rows):
        for c in range(cols):
            atten = np.exp(-decay*((r-rctr)**2 + (c-cctr)**2)**0.5)
            X[:, r, c] = s*atten
    X, N = dtype(X), dtype(N)
    X += N
    return X, N

def load_h5(file_path):
    with h5py.File(file_path, 'r') as f:
        X = f['data'][:]
    return X

def plot_frames(X, timepoints):
    N = len(timepoints)
    fig, axs = plt.subplots(1,N)
    for tidx, to in enumerate(timepoints):
        axs[tidx].imshow(X[to, :, :], aspect=1/20, interpolation=None)
        axs[tidx].set_xticks([])
        axs[tidx].set_yticks([])
    return fig, axs

def cube_to_mat(X):
    X2d = np.reshape(X, (X.shape[0], -1))
    return X2d