import pandas as pd
import os
from matplotlib import pyplot as plt
import numpy as np
from astropy.timeseries import LombScargle
from scipy import signal

class PeriodTechniques():
    # prompt the user to choose the technique
    # not sure what the package includes provide help
    def __init__(self, t, y):
        self.t = t
        self.y = y

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

    def cp(self, fs):
        self.fs = fs
        self.f, self.Pxx_den = signal.periodogram(self.y, self.fs,window='flattop', scaling='spectrum')
        self.period = 2*np.pi/self.w[np.argmax(self.periodogram)]
        return self.period

    def gen_plot(x, y):
        plt.figure(dpi=200)
        plt.plot(x, y, color='black')
        plt.show()
        plt.close()
