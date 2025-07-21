#!usr/bin/python3

import math
import json
import numpy as np
def getMagnitude(cVal):
    """ Return the magnitude of complex variable cVal.
    Note actually returns magnitude ^2 for consistency with 
    pebble implementation!
    """
    #print(cVal)
    sumSq = cVal.real * cVal.real + cVal.imag * cVal.imag
    #print(cVal,"sumSq=%f, abs=%f" % (sumSq, np.abs(cVal)))
    return(sumSq)


def freq2fftBin(freq, freqRes):
    n = int(freq / freqRes)
    return(n)

def getFreqRes(accData, sampleFreq=25.):
    samplePeriod = len(accData)/sampleFreq
    freqRes = 1.0 / samplePeriod
    return(freqRes)


def getRectWin(winLen):
    """ Return a rectangular window of length winLen.
    """
    win = [1.0] * winLen
    winArr = np.array(win)
    return(winArr)

def getRaisedCosineWin(winLen, alpha=0.5):
    """ Return a raised cosine window of length winLen.
    The alpha parameter controls the width of the cosine lobe.
    """
    win = []
    for i in range(winLen):
        win.append(0.5 * (1 - np.cos(2 * np.pi * i / (winLen - 1))))
    winArr = np.array(win)
    return(winArr)

def getHammingWin(winLen):
    """ Return a Hamming window of length winLen.
    """
    win = []
    for i in range(winLen):
        win.append(0.54 - 0.46 * np.cos(2 * np.pi * i / (winLen - 1)))
    winArr = np.array(win)
    return(winArr)  

def getHannWin(winLen):
    """ Return a Hann window of length winLen.
    """
    win = []
    for i in range(winLen):
        win.append(0.5 * (1 - np.cos(2 * np.pi * i / (winLen - 1))))
    winArr = np.array(win)
    return(winArr)

def getTriangularWin(winLen):
    """ Return a triangular window of length winLen.
    """
    win = []
    for i in range(winLen):
        # This is the formulation from wikipedia, the commented version was from github copilot
        win.append(1 - math.fabs((i - winLen/2)/(winLen/2)))
        #if i < winLen / 2:
        #    win.append(2 * i / (winLen - 1))
        #else:
        #    win.append(2 * (winLen - 1 - i) / (winLen - 1))
    winArr = np.array(win)
    return(winArr)


def getWin(winLen, window='rect'):
    if (window=='rect'):
        win = getRectWin(winLen)
    elif (window=='raisedCosine'):
        win = getRaisedCosineWin(winLen)
    elif (window=='hamming'):
        win = getHammingWin(winLen)
    elif (window=='hann'):
        win = getHannWin(winLen)
    elif (window=='triangle'):
        win = getTriangularWin(winLen)
    else:
        print(f"Unrecognised window {window}")
        exit(-1)
    winArr = np.array(win)
    return(winArr)
    

def getFFT(accData, sampleFreq=25, window='rect', debug=False):
    winArr = getWin(len(accData), window)
    accDataWindowed = accData * winArr
    if (debug): print("accData=", accData, accData.shape, accData.dtype)
    if (debug): print("win=", winArr, winArr.shape, winArr.dtype)
    if (debug): print("accDataWindowed = ",accDataWindowed, accDataWindowed.shape, accDataWindowed.dtype)
    fftArr = np.fft.fft(accData)
    fftFreq = np.fft.fftfreq(fftArr.shape[-1], 1.0/sampleFreq)
    if (debug): print("fftArr=", fftArr, fftArr.shape, fftArr.dtype)
    if (debug): print("fftFreq=", fftFreq, fftFreq.shape, fftFreq.dtype)
    return(fftArr, fftFreq)

def getSpecPower(accData, sampleFreq=25, freqCutoff=12.5, plotData = False):
    nSamp = len(accData)
    nFreqCutoff = freq2fftBin(freqCutoff)
    fftArr, fftFreq = getFFT(accData, sampleFreq)
    specPower = 0.
    # range starts at 1 to avoid the DC component
    for i in range(1,nSamp):
        if (i<=nFreqCutoff):
            specPower = specPower + getMagnitude(fftArr[i])
    # The following may not be correct
    specPower = specPower / nSamp / 2
    return specPower


def getRoiPower(accData, sampleFreq=25, alarmFreqMin = 3, alarmFreqMax=8,
                plotData = False):
    nMin = freq2fftBin(alarmFreqMin)
    nMax = freq2fftBin(alarmFreqMax)
    fftArr, fftFreq = getFFT(accData, sampleFreq)
    roiPower = 0.
    for i in range(nMin, nMax):
        roiPower = roiPower + getMagnitude(fftArr[i])
    roiPower = roiPower / (nMax - nMin)
    return roiPower

def getSpectrumRatio(accData, alarmThresh = 900):
    specPower = getSpecPower(accData);
    roiPower = getRoiPower(accData);

    if (specPower > alarmThresh):
        specRatio = 10.0 * roiPower / specPower;
    else:
        specRatio = 0.0;
    return(specRatio);


def getAlarmState(accData, alarmThresh=900, alarmRatioThresh=57):
    ''' getAlarmState(rawData) - determines the alarm state associated with the snapshot of raw
     acceleration data rawData[]
     @return the alarm state (0=ok, 1 = alarm)
    '''
    alarmRatio = getSpectrumRatio(accData);

    if (alarmRatio <= alarmRatioThresh):
        alarmState = 0;
    else:
        alarmState = 1;
    return(alarmState);

                  
if __name__ == "__main__":
    print("libosd.osdAlgTools.__main__")
