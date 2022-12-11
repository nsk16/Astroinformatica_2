import PT
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# test rm_noise
A = 2.0
w0 = 1.0
nin = 150
nout = 1000000
x = np.linspace(0, 10 * np.pi, nin)
n = np.random.normal(scale=1, size=x.size)
y = A * np.sin(w0 * x) + n
w = np.linspace(0.01, 10, nout)
test = PT.PeriodTechniques(x,y)
rm = test.rm_noise(nin)

# test LS
rng = np.random.default_rng()
A = 2.0
w0 = 1.0  # rad/sec
nin = 150
nout = 100000
x = rng.uniform(0, 10 * np.pi, nin)
y1 = A * np.sin(w0 * x)
w = np.linspace(0.01, 10, nout)

#test scipy LS
ts = PT.PeriodTechniques(x,y1)
ls_scipy = ts.lombscargle_scipy(w)
print("Given period:", 2*np.pi/w0)
print("Period from scipy: ",ls_scipy)

#test gatspy Ls
ls_gatspy = ts.lombscargle_gatspy(dy=0.1)
print("Period from gatspy: ", ls_gatspy)

#test cp
ls_cp = ts.cp(fs=0.1592)
print("Period from cp: ", ls_cp[0])

#test PDM
pdm = ts.PDM(2.0, 8.0, 0.1)
print("Period from PDM: ", pdm)

#test Horne1986
t = np.linspace(0, 10 * np.pi, nin)
y = A * np.sin(w0 * t)
ts = PT.PeriodTechniques(t, y)
H_1986 = ts.Horne1986(w)
print(H_1986)
