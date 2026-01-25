#!usr/bin/env python3

import sys
import os
import json
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import libosd
import sdAlg

# ExecuTorch Python runtime APIs
from executorch.runtime import Runtime, Verification

# Use the same preprocessing/buffering as the training implementation
try:
    from user_tools.nnTraining2.deepEpiCnnModel_torch import DeepEpiCnnModelPyTorch
except ImportError:
    # Fallback relative import if running from a different working dir
    from nnTraining2.deepEpiCnnModel_torch import DeepEpiCnnModelPyTorch


class NnAlg(sdAlg.SdAlg):
    """Neural network algorithm using a PyTorch ExecuTorch .pte model.

    Designed to work with .pte models exported from nnTraining2/deepEpiCnnModel.
    Inference runs via ExecuTorch's Python runtime on CPU. Preprocessing uses the
    same buffer logic as training to produce 30s windows (default 750 samples @25Hz).
    """

    def __init__(self, settingsStr, debug=True):
        print("nnAlg.__init__() - settingsStr=%s" % settingsStr)
        print("nnAlg.__init__(): settingsStr=%s (%s)" % (settingsStr, type(settingsStr)))
        super().__init__(settingsStr, debug)

        # Core settings
        self.mModelFname = self.settingsObj['modelFname']
        self.mModeStr = self.settingsObj.get('mode', 'multi')
        self.mSamplePeriod = float(self.settingsObj.get('samplePeriod', 5.0))
        self.mWarnTime = float(self.settingsObj.get('warnTime', 5.0))
        self.mAlarmTime = float(self.settingsObj.get('alarmTime', 10.0))
        self.mNormalise = bool(self.settingsObj.get('normalise', False))
        self.mSdThresh = float(self.settingsObj.get('sdThresh', 5.0))  # stdev % threshold

        # Buffer and sampling config for preprocessing (defaults: 25Hz, 30s)
        self.sampleFreq = float(self.settingsObj.get('sampleFreq', 25.0))
        self.bufferSeconds = float(self.settingsObj.get('bufferSeconds', 30.0))

        # Alarm state
        self.alarmState = 0
        self.alarmCount = 0

        # Preprocessor/buffer from training model implementation
        self.nnModel = DeepEpiCnnModelPyTorch(
            configObj={
                'sampleFreq': self.sampleFreq,
                'bufferSeconds': self.bufferSeconds,
                'convDropout': 0.0,
                'denseDropout': 0.025
            },
            debug=debug,
        )

        # Load ExecuTorch program (.pte)
        self._load_pte_model(self.mModelFname)

    def _load_pte_model(self, pte_path: str):
        """Load ExecuTorch .pte program and forward method."""
        if not os.path.exists(pte_path):
            raise FileNotFoundError(f"PTE model file not found: {pte_path}")

        print(f"nnAlg: Loading ExecuTorch program from {pte_path}")
        rt = Runtime.get()
        program = rt.load_program(pte_path, verification=Verification.Minimal)
        method = program.load_method('forward')
        if method is None:
            # Fallback: pick the first available method
            names = list(program.method_names)
            if len(names) == 0:
                raise RuntimeError("ExecuTorch program contains no methods")
            print(f"nnAlg: 'forward' not found. Using method '{names[0]}' instead.")
            method = program.load_method(names[0])
        self._forward_method = method
        print("nnAlg: ExecuTorch program loaded; ready for inference.")

    def dp2vector(self, dpObj, normalise=False):
        """Convert a datapoint into the buffered input vector for inference.

        Applies a low-movement rejection using `sdThresh` before buffering.
        Returns None until the buffer reaches the required window length.
        """
        if isinstance(dpObj, dict):
            rawDataStr = libosd.dpTools.dp2rawData(dpObj)
        else:
            rawDataStr = dpObj

        accData, hr = libosd.dpTools.getAccelDataFromJson(rawDataStr)
        if accData is None:
            if self.DEBUG:
                print("nnAlg.dp2vector(): No acceleration data in datapoint")
            return None

        # Low-motion rejection based on stdev as percentage of mean
        accArr = np.array(accData, dtype=float)
        accAvg = float(np.average(accArr)) if accArr.size else 0.0
        accStdPct = (100.0 * float(np.std(accArr)) / accAvg) if accAvg != 0 else 0.0
        if accStdPct < self.mSdThresh:
            if self.DEBUG:
                print("nnAlg.dp2vector(): Rejecting low movement datapoint (std% = %.2f < %.2f)" % (accStdPct, self.mSdThresh))
            return None

        # Use training preprocessor buffer to construct window-length vector
        vec_list = self.nnModel.accData2vector(accData, normalise=normalise)
        return vec_list  # list of length ~750 (depending on bufferSeconds*sampleFreq)

    def processDp(self, dpStr, eventId):
        """Process a single datapoint and update alarm state.

        Returns JSON with at least `alarmState`.
        """
        vec = self.dp2vector(dpStr, normalise=self.mNormalise)

        if vec is None:
            inAlarm = False
        else:
            # Prepare input tensor: (1, 1, length)
            x = torch.tensor(np.array(vec, dtype=np.float32)).reshape(1, 1, -1)
            # ExecuTorch method expects a sequence of inputs
            outputs = self._forward_method.execute((x,))
            # outputs is a sequence; take first tensor as logits
            logits = outputs[0]
            probs = torch.softmax(logits, dim=1)
            pSeizure = float(probs[0, 1].item())
            inAlarm = (pSeizure > 0.5)

        if inAlarm:
            self.alarmCount += self.mSamplePeriod
            if self.alarmCount > self.mAlarmTime:
                self.alarmState = 2
            elif self.alarmCount > self.mWarnTime:
                self.alarmState = 1
        else:
            # decay/reset alarm state
            if self.alarmState == 2:
                self.alarmState = 1
                self.alarmCount = self.mWarnTime
            else:
                self.alarmState = 0
                self.alarmCount = 0

        # Single mode forces immediate alarm/no-alarm based solely on current dp
        if self.mModeStr == 'single':
            self.alarmState = 2 if inAlarm else 0

        extraData = {
            'alarmState': self.alarmState,
        }
        return json.dumps(extraData)

    def resetAlg(self):
        self.alarmState = 0
        self.alarmCount = 0
        # Reset the preprocessor buffer at event boundaries
        if hasattr(self.nnModel, 'resetAccBuf'):
            self.nnModel.resetAccBuf()
