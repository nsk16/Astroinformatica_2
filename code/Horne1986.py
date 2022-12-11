import numpy as np

amp = 2.0
w0 = 1.0
nin = 150
nout = 100000
t = np.linspace(0, 10 * np.pi, nin)
y = amp * np.sin(w0 * t)
w = np.linspace(0.01, 10, nout)

P_x = []
for i in w:
    num = sum(np.sin(2 * i * t))
    den = np.cos(2 * i * t)
    tau = np.arctan(num/den)/ (2 * i)
    A = sum((y * np.cos(i*(t-tau))))**2
    B = sum((y * np.sin(i*(t-tau))))**2
    C = sum(np.cos(i*(t - tau))**2)
    D = sum(np.sin(i*(t - tau))**2)
    P = (1/2 * (A/C + B/D)) / (np.std(y))**2
    P_x.append(P)
print(2*np.pi/w[np.argmax(P_x)])
