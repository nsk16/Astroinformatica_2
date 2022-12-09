from scipy import fftpack
from matplotlib import pyplot as plt
import numpy as np

# to see if the signal is periodic and to clean the signal with noise and determine a rough estimate of period
rng = np.random.default_rng()

A = 2.0
w0 = 1.0
nin = 150
nout = 1000000
x = np.linspace(0, 10 * np.pi, 1000)
y = A * np.sin(w0 * x)
w = np.linspace(0.01, 10, nout)

rng = np.random.default_rng()
A = 2.0
w0 = 1.0
nin = 150
nout = 1000000
x = rng.uniform(0, 10 * np.pi, nin)
y = A * np.sin(w0 * x)
w = np.linspace(0.01, 10, nout)
sig_fft = fftpack.fft(y)
amp = np.abs(sig_fft)
power = amp**2
phase = np.angle(sig_fft)
freq = fftpack.fftfreq(y.size, 1/1000)
period = 2 * np.pi/freq[np.argmax(power)]
plt.plot(freq, power)
plt.show()
print(period)
