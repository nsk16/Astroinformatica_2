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
#in this case period = 2*pi/frequency(w0)
# visualize
# plt.figure(figsize=(7,7),dpi=100)
# plt.scatter(x,y, color='black')
# plt.xlabel('Time (s)')
# # plt.ylabel('power')
# plt.savefig(fig_path+'dummy_data_LS_test.pdf' ,dpi=200, facecolor='w')
# plt.show()

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
model = LombScargleFast().fit(x,y, dy=0.1)
periods, power = model.periodogram_auto(nyquist_factor=100)
print("Results gatspy:")
print("period: {} s ".format(periods[np.argmax(power)]))
exit()
#plot
fig, axs = plt.subplots(4,1, figsize=(10, 10), facecolor='w')
axs = axs.ravel()
axs[0].set_title("sine Function", color='blue', fontweight='bold')
axs[0].scatter(x, y, marker='.', color='black')
axs[0].set(xlabel='Time (s)')
axs[1].plot(freq, power1, color='green')
axs[1].set(xlabel='Frequency (1/s)', ylabel='Lomb-Scargle power')
axs[2].plot(w, p, color='green')
axs[2].set(xlabel='Frequency (rad/s)', ylabel='Lomb-Scargle power')
axs[3].plot(periods, power, color='green')
axs[1].set(xlabel='Frequency (1/s)', ylabel='Lomb-Scargle power')
plt.show()

#fft
p = np.fft.fft(y)
freq = np.fft.fftfreq(x.shape[-1])
plt.plot(freq, abs(p))
plt.show()
exit()
# test with real data
data = pd.read_csv('33836115_sector01_4_2_cleaned.lc',sep="[ ,]",engine = "python",na_values=['*********','9.999999', '********', 'NaN'],names = ['time', 'mag','err'],usecols = [0,1,2])
# print(data)
t, mag, dmag = LINEAR_data.get_light_curve('10040133').T
t = t *u.d

model = LombScargleFast().fit(t, mag, dmag)
periods, power = model.periodogram_auto(nyquist_factor=100)

fig, ax = plt.subplots()
ax.plot(periods, power)
ax.set(xlim=(0,1.4),xlabel='period(days)', ylabel='Lomb-Scargle Power')
plt.show()
model.optimizer.period_range=(0.2, 1.4)
period = model.best_period
print("period = {0}".format(period))

t = data['time']
y = data['mag']
dy = data['err']

plt.figure(figsize=(7,7),dpi=100)
plt.scatter(t, y, color='black')
plt.xlabel('Time (s)')
plt.ylabel('mag')
# plt.savefig(fig_path+'dummy_data_LS_test.pdf' ,dpi=200, facecolor='w')
plt.show()
#test astropy

freq, power = LombScargle(t, mag).autopower()
# ls = LombScargle(t, mag, dmag)
# print(freq.unit)

print('Results from LS-astropy:')
print('maximum power: ',power.max())
# print('false alarm probability: ',ls.false_alarm_probability(power.max()))
print('Freq at max power {} '.format(freq[np.argmax(power)]))
print('Period at max. power: {} s'.format(1/freq[np.argmax(power)]))
plt.figure(figsize=(7,7),dpi=100)
plt.scatter(freq, power, color='black')
plt.xlim(0,0.35)
plt.xlabel('Time (s)')
plt.ylabel('mag')
# plt.savefig(fig_path+'dummy_data_LS_test.pdf' ,dpi=200, facecolor='w')
plt.show()
exit()
#test scipy
# w = np.linspace(-2*np.pi, 2*np.pi, 100000)
period = np.linspace(0.2, 1.4, 4000)
omega = 2 * np.pi / period
p = signal.lombscargle(t, mag-mag.mean(), omega)
N = len(t)
p *= 2/ (N*mag.std()**2)
 # w = 2 pi nu
print('Results from LS-scipy:')
print('maximum power: ',p.max())
print('Angular freq at max power {} '.format(w[np.argmax(p)]))
print('Period at max. power: {} s'.format(2*np.pi/omega[np.argmax(p)]))
plt.figure(figsize=(7,7),dpi=100)
plt.scatter(period, p, color='black')
# plt.xlim(6, 20)
plt.xlabel('Time (s)')
plt.ylabel('mag')
# plt.savefig(fig_path+'dummy_data_LS_test.pdf' ,dpi=200, facecolor='w')
plt.show()
# model = LombScargleFast().fit(t, mag)
# # model.optimizer.period_range = (1.0, 7.0)
# periods, power = model.periodogram_auto(nyquist_factor=100)
# print('Results from LS-gatspy:')
# print('Period at max. power: {} s'.format(periods[np.argmax(power)]))
