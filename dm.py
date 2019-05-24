import numpy as np
import pylab as plt
import argparse

# A super quick script to show the transfer
# function for cold plasma dispersion

# Read some command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-tobs', type=float, dest='tobs', help='set the duration of the file in seconds (default: 1.0)', default=1.0)
parser.add_argument('-dm', type=float, dest='dm', help='set the dispersion measure of the signal in pc/cc (default: zero)', default=2.0)
parser.add_argument('-fcentre', type=float, dest='fcentre', help='set the centre frequency in MHz (default: 843.0)', default=1400.0)
parser.add_argument('-bw', type=float, dest='bw', help='set the bandwidth in MHz (default: 400.0)', default=32.0)
args = parser.parse_args()

# Input numbers
dm      = args.dm                             # in pc/cc 
fcentre = args.fcentre                        # in MHz
bw      = args.bw                             # in MHz

# Calculate some numbers based on inputs
tsamp   = 1.0e-6/(2.0*bw)                     # in seconds (real-sampled data)
ftop    = fcentre + 0.5*bw                    # in MHz
fbot    = fcentre - 0.5*bw                    # in MHz
dd      = 2.0*np.pi*4149.0*1.0e6*dm           # 
tDM     = dd*1.0e-6*(fbot**(-2) - ftop**(-2)) # in seconds
nchans  = 2**int(np.log(int(np.floor(tDM/tsamp)))/np.log(2)+1) # choose number of FFT channels to be rounded up to next highest power of 2 needed to cover the DM sweep in the band
print nchans
foff    = bw/nchans                           # in MHz
taper   = 1.0                                 # with no taper you get some batman ears

H = np.zeros(nchans+1,dtype=np.complex_)
for k in range(0,nchans+1):
    f = fbot + k*foff - fcentre
#    taper = 1.0/np.sqrt(1.0+(f/(0.47*bw))**80)
    fac = dd*f**2/((f+fcentre)*(fcentre**2))
    H[k] = np.cos(fac) + np.sin(fac)*1j
    H[k] = taper*H[k]
#    print k,H[k]

plt.plot(H.real)
plt.plot(H.imag)
plt.show()

h = np.fft.fftshift(np.fft.fft(H))
plt.plot(np.abs(h))
plt.show()

# NEXT STEP
# make an array of noise + signals of length Nsamps=Tobs/tsamp
# In chunks of size np.size(H) do chunk_d = fftshift(irfft(rfft(chunk)*H))
# For multiple chunks need to look at overlap-save stuff
# 
