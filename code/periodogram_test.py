import pandas as pd
import os
from matplotlib import pyplot as plt
import numpy as np
from astropy.timeseries import LombScargle
from scipy import signal

rng = np.random.default_rng()
A = 2.0

w0 = 1.0  # rad/sec

nin = 150

nout = 100000
x = rng.uniform(0, 10 * np.pi, nin)
y = A * np.cos(w0 * x)
w = np.linspace(0.01, 10, nout)
pgram = signal.lombscargle(x, y, w, normalize=True)

fig, (ax_t, ax_w) = plt.subplots(2, 1, constrained_layout=True)
ax_t.plot(x, y, "b+")
ax_t.set_xlabel("Time [s]")
ax_w.plot(w, pgram)
ax_w.set_xlabel("Angular frequency [rad/s]")
ax_w.set_ylabel("Normalized amplitude")
plt.show()

print("maximum power: ", pgram.max())
print("Angular freq at max power {} ".format(w[np.argmax(pgram)]))
print("Period at max. power: {} s".format(2 * np.pi / w[np.argmax(pgram)]))
#
# path = 'test_data/'
# fig_path = 'test_fig/'
# data1 = pd.read_csv(path+'41259805_sector01_4_2.lc',sep="[ ,]",engine = "python",na_values=['*********','9.999999', '********', 'NaN'],names = ['time', 'mag','err'],usecols = [0,1,3])
#
# print(data1.head())
# # plt.figure(figsize=(7,7),dpi=100)
# # plt.plot(data1['time'],data1['mag'],color='black')
# # plt.xlabel('t (MJD)')
# # plt.ylabel('mag')
# # plt.savefig(fig_path + 'data.pdf' ,dpi=200,facecolor='w')
# # plt.show()
# # plt.close()
#
# # Lombscargle astropy
# t = data1['time']
# y = data1['mag']
# dy = data1['err']
# freq, power = LombScargle(t,y, dy).autopower()
# ls = LombScargle(t, y, dy)
# plt.figure(figsize=(7,7),dpi=100)
# plt.plot(freq, power)
# plt.xlabel(r'frequency (d$^{-1}$)')
# plt.ylabel('power')
# plt.xlim(0,2)
# plt.savefig(fig_path+'LS_astropy.pdf' ,dpi=200, facecolor='w')
# plt.show()
# plt.close()
# # print(len(t))
# print('Results from LS-astropy:')
# # print('maximum power: ',power.max())
# # print(np.argmax(power))
# # print('false alarm probability: ',ls.false_alarm_probability(power.max()))
# # print('Freq at max power {} '.format(freq[np.argmax(power)]))
# print('Period at max. power: {} s'.format(86400/freq[np.argmax(power)]))
#
# # LombScargle scipy
# w = np.linspace(-2, 2, 3000)
# p = signal.lombscargle(t, y, w) # w = 2 pi nu
# plt.figure(figsize=(7,7),dpi=100)
# plt.plot(w, p)
# plt.xlabel(r'$\omega$ (rad s$^{-1}$)')
# plt.ylabel('Amplitude')
# plt.savefig(fig_path+'LS_scipy.pdf' ,dpi=200, facecolor='w')
# plt.show()
# plt.close()
#
# print('Results from LS-scipy:')
# # print('maximum power: ',p.max())
# # print('Angular freq at max power {} '.format(w[np.argmax(p)]))
# print('Period at max. power: {} s'.format(2*np.pi/w[np.argmax(p)]))
#
# # classical periodogram
# f, Pxx_den = signal.periodogram(y, window='flattop', scaling='spectrum')
# plt.figure(figsize=(7,7),dpi=100)
# plt.plot(f, Pxx_den)
# plt.xlim(0, 0.1)
# plt.xlabel(r'$ Frequency$ (s$^{-1}$)')
# plt.ylabel('Power spectral density')
# plt.savefig(fig_path+'cp.pdf' ,dpi=200, facecolor='w')
# plt.show()
# plt.close()
#
# print('Results from classical periodogram:')
# # print('maximum power: ',.max())
# # print('Angular freq at max power {} '.format(w[np.argmax(p)]))
# print('Period at max. power: {} s'.format(1/f[np.argmax(Pxx_den)]))
#
#
# # max_w = w[np.argmax(p)]
# # max_f = max_w/(2*np.pi)
# # Period = 2*np.pi/max_w
