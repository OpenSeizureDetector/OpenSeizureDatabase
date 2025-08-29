#!/usr/bin/env python3

import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..','..'))
#import libosd.webApiConnection
import libosd.osdAlgTools as oat

accDataLen = 125

accDataLst = [ 1000 ] * accDataLen

print(accDataLst)

fftArr, fftFreq = oat.getFFT(accDataLst, sampleFreq=25, window='rect')


print(fftArr, fftFreq)
