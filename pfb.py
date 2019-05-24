"""
# pfb.py

Simple implementation of a polyphase filterbank.
"""
import numpy as np
import scipy
from scipy.signal import firwin, freqz, lfilter


def db(x):
    """ Convert linear value to dB value """
    return 10*np.log10(x)

def generate_win_coeffs(M, P, window_fn="hamming"):
    win_coeffs = scipy.signal.get_window(window_fn, M*P)
    sinc       = scipy.signal.firwin(M * P, cutoff=1.0/P, window="rectangular")
    win_coeffs *= sinc
    return win_coeffs

def pfb_fir_frontend(x, win_coeffs, M, P):
    W = x.shape[0] / M / P
    x_p = x.reshape((W*M, P)).T
    h_p = win_coeffs.reshape((M, P)).T
    x_summed = np.zeros((P, M * W - M))
    for t in range(0, M*W-M):
        x_weighted = x_p[:, t:t+M] * h_p
        x_summed[:, t] = x_weighted.sum(axis=1)
    return x_summed.T

def fft(x_p, P, axis=1):
    return np.fft.rfft(x_p, P, axis=axis)

def pfb_filterbank(x, win_coeffs, M, P):
    x_fir = pfb_fir_frontend(x, win_coeffs, M, P)
    x_pfb = fft(x_fir, P)

def pfb_spectrometer(x, n_taps, n_chan, n_int, window_fn="hamming"):
    M = n_taps
    P = n_chan
    
    # Generate window coefficients
    win_coeffs = generate_win_coeffs(M, P, window_fn)

    # Apply frontend, take FFT, then take power (i.e. square)
    x_fir = pfb_fir_frontend(x, win_coeffs, M, P)
    x_pfb = fft(x_fir, P)
    x_psd = np.abs(x_pfb)**2
    
    # Trim array so we can do time integration
    x_psd = x_psd[:np.round(x_psd.shape[0]/n_int)*n_int]
    
    # Integrate over time, by reshaping and summing over axis (efficient)
    x_psd = x_psd.reshape(x_psd.shape[0]/n_int, n_int, x_psd.shape[1])
    x_psd = x_psd.mean(axis=1)
    
    return x_psd

if __name__ == "__main__":
    import pylab as plt
    import argparse
    import seaborn as sns
    sns.set_style("white")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-ntaps', type=int, dest='ntaps', help='set the number of FIR filter taps in the PFB (default: 4)', default=4)
    parser.add_argument('-nchans', type=int, dest='nchans', help='set the number of output channels from the PFB (default: 1024)', default=1024)
    parser.add_argument('-tobs', type=float, dest='tobs', help='set the duration of the file in seconds (default: 1.0)', default=1.0)
    parser.add_argument('-bw', type=float, dest='bw', help='set the bandwidth in MHz (default: 400.0)', default=400.0)
    parser.add_argument('-fcentre', type=float, dest='fcentre', help='set the centre frequency in MHz (default: 1400.0)', default=1400.0)
    args = parser.parse_args()

    M      = args.ntaps            # Number of taps
    P      = 2*args.nchans         # Number of 'branches', also fft length
    Tobs   = args.tobs             # Tobs in seconds
    nsamps = Tobs*(2.0*args.bw*1.0e6)  # Number of Nyquist samples in Tobs
    W      = int(nsamps/(M*P))     # Number of windows in Tobs of complete M*P size chunks
#    print M, P, W

    # Generate a test data steam
    samples = np.arange(M*P*W)
    noise   = np.random.random(M*P*W) 
    freq    = 1
    amp     = 1
    cw_signal = amp * np.sin(samples * freq)
    data = noise + cw_signal
    
    X_psd = pfb_spectrometer(data, n_taps=M, n_chan=P, n_int=2, window_fn="hamming")

    plt.imshow(db(X_psd), cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.xlabel("Channel")
    plt.ylabel("Time")
    
    plt.show()
    
