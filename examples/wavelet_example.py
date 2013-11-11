from sdppy.wavelet import wavelet
import numpy as np
import matplotlib.pyplot as plt

# points per period
amplitude = 1
ppp = 40
period = 2 * np.pi
sampling = period / ppp
# 180 seconds, base timing
t_period = 180
dt = t_period / ppp
t_total = 245 * 1
t_samples = int(t_total / dt)

t = np.array(range(-t_samples, t_samples, 1), dtype='f4')
time = t * sampling * t_period
# base sinusoid
t1 = time / t_period
x1 = np.sin(t1) * amplitude
# 5-min period
t2 = time / (5 * 60)
x2 = np.sin(t2) * amplitude

t3 = time / (13 * 60)
x3 = np.sin(t3) * amplitude
# total signal
x = x1 + x2 + x3

w = wavelet(x, dt=dt, pad=False, core='morlet')
recon = w.y1
dif = x - w.y1
print(np.max(dif))

plt.figure(1)
plt.plot(x, label='Original')
plt.plot(w.y1, label='Reconstruction')
plt.legend()
plt.figure(3)
plt.plot(dif)
plt.show()
