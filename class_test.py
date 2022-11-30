import class_base
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


rng = np.random.default_rng()
A = 2.0
w0 = 1.0  # rad/sec
nin = 150
nout = 100000
x = rng.uniform(0, 10 * np.pi, nin)
y = A * np.sin(w0 * x)
w = np.linspace(0.01, 10, nout)

ts = class_base.PeriodTechniques(x,y)
ls_scipy = ts.lombscargle_scipy(w)
print("Period from scipy: ",ls_scipy)
ls_astropy = ts.lombscargle_astropy(dy=0.1)
print("Period from astropy: ", ls_astropy[0])
ls_cp = ts.cp(fs=0.1592)
print("Period from cp: ", ls_cp)

# rand = np.random.default_rng(42)
# t1 = 100 * rand.random(100)
# y1 = np.sin(2 * np.pi * t1) + 0.1 * rand.standard_normal(100)
# ts = class_base.PeriodTechniques(t1,y1)
# ls_scipy = ts.lombscargle_scipy(w)
# print("Period from scipy: ",ls_scipy[0])
# ls_astropy = ts.lombscargle_astropy(dy=0.1)
# print("Period from astropy: ", ls_astropy)
# ls_cp = ts.cp(fs=0.1592)
# print("Period from cp: ", ls_cp)

plt.figure(figsize=(7,7),dpi=100)
plt.plot(ls_astropy[1], ls_astropy[2], color='black')
plt.plot(w ,ls_scipy[1], color='red')
plt.xlabel(r'frequency (d$^{-1}$)')
plt.ylabel('power')
# plt.xlim(0,2)
# plt.savefig(fig_path+'LS_astropy.pdf' ,dpi=200, facecolor='w')
plt.show()
