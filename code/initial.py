from scipy import fftpack
from matplotlib import pyplot as plt
import numpy as np

# to see if the signal is periodic and to clean the signal with noise and determine a rough estimate of period
rng = np.random.default_rng()

A = 2.0
w0 = 1.0
nin = 150
nout = 1000000
x = np.linspace(0, 10 * np.pi, nin)
n = np.random.normal(scale=1, size=x.size)

y = A * np.sin(w0 * x) + n
w = np.linspace(0.01, 10, nout)

plt.plot(x,y)
plt.show()

sig_fft = fftpack.fft(y)
amp = np.abs(sig_fft)
power = amp**2
phase = np.angle(sig_fft)
freq = fftpack.fftfreq(y.size, 1/nin)
period = 2 * np.pi/freq[np.argmax(power)]
amp_freq = np.array([amp, freq])
position = amp_freq[0,:].argmax()
peak_f = amp_freq[1, position]
plt.plot(freq, power)
plt.show()
# removing noise
sig_fft[np.abs(freq) > peak_f ] = 0

sig_rm_noise = fftpack.ifft(sig_fft)
plt.plot(x,y)
plt.plot(x,sig_rm_noise)
plt.show()
# print(period)
