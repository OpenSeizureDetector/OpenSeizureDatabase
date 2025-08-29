import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

print(sys.path)

import libosd.osdAlgTools as osdAlgTools

print (dir(osdAlgTools))

# python


class TestOsdAlgTools(unittest.TestCase):

    def test_getTriangularWin(self):
        winLen = 10
        win = osdAlgTools.getTriangularWin(winLen)
        self.assertEqual(win.shape[0], winLen)
        self.assertTrue(np.all(win >= 0))
        self.assertTrue(np.all(win <= 1))
        self.assertTrue(win[5] == 1)  # Center value should be 1 for even length

    def test_getHammingWin(self):
        winLen = 10
        win = osdAlgTools.getHammingWin(winLen)
        self.assertEqual(win.shape[0], winLen)
        self.assertTrue(np.all(win >= 0))
        self.assertTrue(np.all(win <= 1))
        self.assertAlmostEqual(win[0], 0.08, places=2)  # First value should be close to 0.08 for a Hamming window of length 10
        self.assertAlmostEqual(win[winLen - 1], 0.08, places=2)  # Last value should be close to 0.08 for a Hamming window of length 10
        self

    def test_getHannWin(self):
        winLen = 10
        win = osdAlgTools.getHannWin(winLen)
        self.assertEqual(win.shape[0], winLen)
        self.assertTrue(np.all(win >= 0))
        self.assertTrue(np.all(win <= 1))
        self.assertAlmostEqual(win[0], 0.0,  places=2)  # First value should be close to 0.0 for a Hann window of length 10
        self.assertAlmostEqual(win[winLen - 1], 0.0, places=2)  # Last value should be close to 0.0 for a Hann window of length 10
        self

    def test_getRectWin(self):
        winLen = 10
        win = osdAlgTools.getRectWin(winLen)
        self.assertEqual(win.shape[0], winLen)
        self.assertTrue(np.all(win == 1))

    def test_getRaisedCosineWin(self):
        winLen = 10
        win = osdAlgTools.getRaisedCosineWin(winLen)
        self.assertEqual(win.shape[0], winLen)
        self.assertTrue(np.all(win >= 0))
        self.assertTrue(np.all(win <= 1))
        self.assertAlmostEqual(win[0], 0., places=2) # First value should be close to 0 for a raised cosine window of length 10
        self.assertAlmostEqual(win[winLen // 2], 1.0, places= 1) # Middle value should be 1.0 for a raised cosine window of length 10
        self.assertAlmostEqual(win[winLen - 1], 0., places= 2) # Last value should be close to 0. for a raised cosine window of length 10


def plotWindowShapes():
    import matplotlib.pyplot as plt

    winLen = 125
    windows = {
        'Rectangular': osdAlgTools.getRectWin(winLen),
        'Raised Cosine': osdAlgTools.getRaisedCosineWin(winLen),
        'Hamming': osdAlgTools.getHammingWin(winLen),
        'Hann': osdAlgTools.getHannWin(winLen),
        'Triangular': osdAlgTools.getTriangularWin(winLen)
    }

    plt.figure(figsize=(12, 8))
    for name, win in windows.items():
        plt.plot(win, label=name)

    plt.title('Window Shapes')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid()
    plt.show()



if __name__ == "__main__":
    plotWindowShapes()
    unittest.main()
