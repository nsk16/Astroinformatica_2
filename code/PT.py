import pandas as pd
import os
from matplotlib import pyplot as plt
import numpy as np
from scipy import signal
from scipy import fftpack
from gatspy.periodic import LombScargleFast
from PyAstronomy.pyTiming import pyPDM

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
        self.filename = input("Enter the name of the figure to be saved: ")
        plt.savefig(self.filename+'.pdf', dpi=200,facecolor='w')
        plt.show()
        plt.close()

    def cp(self, fs=1.0):
        self.fs = fs
        self.f, self.Pxx_spec = signal.periodogram(self.y, self.fs,window='flattop', scaling='spectrum')
        self.period = 1/self.f[np.argmax(self.Pxx_spec)]
        amp = np.sqrt(self.Pxx_spec.max())
        return self.period, amp

    def lombscargle_gatspy(self, dy):
        self.dy = dy
        model = LombScargleFast().fit(self.t, self.y, dy=self.dy)
        periods, power = model.periodogram_auto(nyquist_factor=100)
        self.period = periods[np.argmax(power)]
        return self.period

    def lombscargle_scipy(self, w):
        self.w = w
        self.periodogram = signal.lombscargle(self.t, self.y, self.w, normalize=True)
        self.period = 2*np.pi/self.w[np.argmax(self.periodogram)] # in units of sec
        return self.period, self.periodogram

    def PDM(self, min, max, dval):
        self.min = min
        self.max = max
        P = pyPDM.PyPDM(self.t, self.y)
        scanner = pyPDM.Scanner(minVal=min, maxVal=max, dVal=dval, mode="period")
        self.periods, self.theta = P.pdmEquiBinCover(10, 3, scanner)
        self.period = self.periods[np.argmin(self.theta)]
        return self.period # period = min of theta

    def rm_noise(self, step_size):
        self.step_size = step_size
        sig_fft = fftpack.fft(self.y)
        amp = np.abs(sig_fft)
        freq = fftpack.fftfreq(self.y.size, 1/self.step_size)
        amp_freq = np.array([amp, freq])
        position = amp_freq[0,:].argmax()
        peak_f = amp_freq[1, position]
        sig_fft[np.abs(freq) > peak_f ] = 0
        self.sig_rm_noise = fftpack.ifft(sig_fft)
        plt.figure(dpi=200)
        plt.plot(self.t,self.y, marker='.', color='blue', alpha=0.5, label='raw data')
        plt.plot(self.t,self.sig_rm_noise, color='orange', label = 'noise removed')
        plt.xlabel('Time ({})'.format(self.t_unit))
        plt.ylabel('{}'.format(self.y_unit))
        plt.legend()
        plt.savefig(self.filename+'_rm_noise.pdf', dpi=200,facecolor='w')
        plt.legend()
        plt.show()
        return self.sig_rm_noise

    def Horne1986(self, w):
        self.w = w
        P_x = []
        for i in self.w:
            num = sum(np.sin(2 * i * self.t))
            den = np.cos(2 * i * self.t)
            tau = np.arctan(num/den)/ (2 * i)
            A = sum((self.y * np.cos(i*(self.t-tau))))**2
            B = sum((self.y * np.sin(i*(self.t-tau))))**2
            C = sum(np.cos(i*(self.t - tau))**2)
            D = sum(np.sin(i*(self.t - tau))**2)
            P = (1/2 * (A/C + B/D)) / (np.std(self.y))**2
            P_x.append(P)
        self.period = 2*np.pi/self.w[np.argmax(P_x)]
        return self.period
