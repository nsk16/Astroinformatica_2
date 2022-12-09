import numpy as np


rng = np.random.default_rng()
A = 2.0
w0 = 1.0
nin = 150
nout = 100000
t = rng.uniform(0, 10 * np.pi, nin)
y = A * np.sin(w0 * t)
w = np.linspace(0.01, 10, nin)
print(w.type)
exit()
tau = np.arctan(np.sin(2 * w * t)/np.cos(2 * w * t))/(2*w)

P = 1/2 * ((((y) * np.cos(w(t-tau)))**2/(np.cos(w(t - tau)))**2) + (((y) * np.sin(w(t-tau)))**2/(np.sin(w(t - tau)))**2))
