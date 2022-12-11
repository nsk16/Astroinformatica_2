import pandas as pd
import os
from matplotlib import pyplot as plt
import numpy as np
from astropy.timeseries import LombScargle
from scipy import signal
from gatspy.periodic import LombScargleFast
from astropy import units as u
import timeit
from gatspy import datasets, periodic
from astroML.time_series import lomb_scargle
from astropy.utils.data import get_pkg_data_filename
from astroML.datasets import fetch_LINEAR_sample
from scipy import fftpack
LINEAR_data = fetch_LINEAR_sample()
#define path
fig_path = '../test_fig/'

#dummy time timeseries
rng = np.random.default_rng()
A = 2.0
w0 = 1.0
nin = 150
nout = 100000
x = rng.uniform(0, 10 * np.pi, nin)
y = A * np.sin(w0 * x)
w = np.linspace(0.01, 10, nout)


#test astropy
freq, power1 = LombScargle(x,y, dy=0.1).autopower()
ls = LombScargle(x, y, dy=0.1)
print('Results from LS-astropy:')
print('maximum power: ',power1.max())
print('false alarm probability: ',ls.false_alarm_probability(power1.max()))
print('Freq at max power {} '.format(freq[np.argmax(power1)]))
print('Period at max. power: {} s'.format(1/freq[np.argmax(power1)]))


#test scipy
p = signal.lombscargle(x, y, w, normalize=True) # w = 2 pi nu
print('Results from LS-scipy:')
print('maximum power: ',p.max())
print('Angular freq at max power {} '.format(w[np.argmax(p)]))
print('Period at max. power: {} s'.format(2*np.pi/w[np.argmax(p)]))

#test gatspy
model = LombScargleFast().fit(x, y, dy=0.1)
periods, power = model.periodogram_auto(nyquist_factor=100)
print("Results gatspy:")
print("Period: {} s ".format(periods[np.argmax(power)]))

#test cp
f, Pxx_spec = signal.periodogram(y, fs = 1.0, window='flattop', scaling='spectrum')
period = 1/f[np.argmax(Pxx_spec)]
print("CP: {} s".format(period))
