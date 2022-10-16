#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import math

timArr = []
accSim = []
magSim = []

ampl = 1200.  # mg
g = 1000.   # mg
sampleFreq = 25.  # Hz

freq = 5.  # Hz

for i in range(0,125):
    t = i/sampleFreq   # time (sec)
    a = g + ampl * math.sin(2*np.pi*t*freq)
    timArr.append(t)
    accSim.append(a)
    magSim.append(math.sqrt(a*a + g*g + 0*0))

#Calculate FFT to see what we have:
fftArr = np.fft.fft(accSim)
fftMagArr = np.fft.fft(magSim)
fftFreq = np.fft.fftfreq(fftArr.shape[-1], 1.0/sampleFreq)

fftReal = []
for cVal in fftArr:
    fftReal.append(math.sqrt(cVal.real * cVal.real + cVal.imag * cVal.imag))

fftMagReal = []
for cVal in fftMagArr:
    fftMagReal.append(cVal.real * cVal.real + cVal.imag * cVal.imag)

# FIXME - zeroing out DC signal to avoid it messing up the graphs - should
# really scale the graphs properly!
fftReal[0] = 0.
fftMagReal[0] = 0.

fig, ax = plt.subplots(2,2)
fig.suptitle('title' % (),
                     fontsize=11)
ax[0][0].plot(timArr, accSim)
ax[0][0].set_title("Simulated Data")
ax[0][0].set_ylabel("Acceleration (~milli-g)")
ax[0][0].grid(True)
ax[1][0].plot(fftFreq, fftReal)
ax[1][0].set_title("FFT - from accel")
ax[1][0].set_ylabel("Power")
ax[1][0].set_xlabel("Freq (Hz)")
ax[1][0].set_xlim(0,15)
ax[1][0].grid(True)
ax[0][1].plot(timArr, magSim)
ax[0][1].set_title("magnitude")
ax[0][1].set_ylabel("Power")
ax[0][1].set_xlabel("time(s)")
ax[0][1].grid(True)
ax[1][1].plot(fftFreq, fftMagReal)
ax[1][1].set_title("FFT - from mag")
ax[1][1].set_ylabel("Power")
ax[1][1].set_xlabel("Freq (Hz)")
ax[1][1].set_xlim(0,15)
ax[1][1].grid(True)
fig.tight_layout()
fig.subplots_adjust(top=0.85)
fig.savefig("sim.png")
plt.close(fig)



