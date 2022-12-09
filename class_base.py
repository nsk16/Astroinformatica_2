import pandas as pd
import os
from matplotlib import pyplot as plt
import numpy as np
from astropy.timeseries import LombScargle
from scipy import signal

class PeriodTechniques():

    def __init__(self, t, y):
        self.t = t
        self.y = y
        self.t_unit = input("Enter the unit of time: ")
        self.y_unit = input("Enter the unit of y: ")
        plt.figure(dpi=200)
        plt.scatter(t, y, color='black', marker='.')
        plt.xlabel('Time ({})'.format(self.t_unit))
        plt.ylabel('{}'.format(self.y_unit))
        plt.savefig('data.pdf', dpi=200,facecolor='w')
        plt.show()
        plt.close()



    def cp(self, fs=1.0):
        self.fs = fs
        self.f, self.Pxx_spec = signal.periodogram(self.y, self.fs,window='flattop', scaling='spectrum')
        self.period = 1/self.f[np.argmax(self.Pxx_spec)]
        amp = np.sqrt(self.Pxx_spec.max())
        return self.period, amp

    def lombscargle_astropy(self, dy):
        self.dy = dy
        self.freq, self.power = LombScargle(self.t, self.y, self.dy, normalization='log').autopower()
        self.period = 1/self.freq[np.argmax(self.power)] #in units of x passed
        return self.period, self.freq, self.power

    def lombscargle_scipy(self, w):
        self.w = w
        self.periodogram = signal.lombscargle(self.t, self.y, self.w, normalize=True)
        self.period = 2*np.pi/self.w[np.argmax(self.periodogram)] # in units of sec
        return self.period, self.periodogram



    def gen_plot(x, y):
        plt.figure(dpi=200)
        plt.plot(x, y, color='black')
        plt.show()
        plt.close()
