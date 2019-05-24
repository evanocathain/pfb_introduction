import numpy as np
import pylab as plt

# A super quick script to show the transfer
# function for cold plasma dispersion

dm = 1000.0
fcentre = 1400.0
bw      = 400.0
nchans  = 1024
foff    = bw/nchans
fbot    = fcentre - foff*nchans/2
ftop    = fcentre + foff*nchans/2
dd      = 2.0*3.14159*4149.0*1.0e6*dm 

H = np.zeros(nchans+1,dtype=np.complex_)
for k in range(0,nchans+1):
    f = fbot + k*foff - fcentre
    fac = dd*f**2/((f+fcentre)*(fcentre**2))
    H[k] = np.cos(fac) + np.sin(fac)*1j
    print k,H[k]

plt.plot(H.real)
plt.plot(H.imag)
plt.show()

