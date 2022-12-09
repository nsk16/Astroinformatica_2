import numpy as np
import matplotlib.pylab as plt
from PyAstronomy.pyTiming import pyPDM

rng = np.random.default_rng()
A = 2.0
w0 = 1.0
nin = 150
nout = 100000
x = rng.uniform(0, 10 * np.pi, nin)
y = A * np.sin(w0 * x)
w = np.linspace(0.01, 10, nout)
P = pyPDM.PyPDM(x, y)

scanner = pyPDM.Scanner(minVal=2.0, maxVal=8.0, dVal=0.05, mode="period")
f1, t1 = P.pdmEquiBinCover(10, 3, scanner)
print(f1[np.argmin(t1)])
exit()
print("Periods: ", end=' ')
for period in scanner:
    print(period, end=' ')

# Show the result
plt.figure(facecolor='white')
plt.title("Result of PDM analysis")
plt.xlabel("Frequency")
plt.ylabel("Theta")
plt.plot(f1, t1, 'bp-')
# plt.plot(f2, t2, 'gp-')
plt.legend(["pdmEquiBinCover", "pdmEquiBin"])
plt.show()
