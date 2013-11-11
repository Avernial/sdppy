from sdppy.emd import emd
from sdppy.specf import imfhspectrum
import numpy as np
import matplotlib.pyplot as plt

t = np.array(range(0, 314, 1)) * 0.5
x = np.sin(t) + np.sin(t / 8)

result = emd(x, shiftfactor=3, interp_kind='cubic')

print("The number of modes", len(result))
# EMD
plt.figure(1)
plt.suptitle('EMD')
plt.subplot(211)
for num, imf in enumerate(result):
    plt.plot(imf, label='{0} imf'.format(num))
plt.axis('tight')
plt.legend()

plt.subplot(212)
plt.grid(True)
plt.title('Summary')
print("Summary")
plt.plot(result.get_recon(), label='Recon')
plt.plot(x, label='Original')
plt.legend(loc='best')

plt.figure(2)
plt.title('Hilbert spectrum')
spec, freq = result.get_hspectr()
plt.imshow(spec)
plt.axis('tight')
plt.figure(3)
plt.title("Power")
plt.plot(sum(spec, 0))
plt.axis('tight')

plt.show()
