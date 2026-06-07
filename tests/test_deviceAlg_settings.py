#!/usr/bin/env python

import json
import os
import sys
import unittest
from unittest.mock import patch

# Mirror other tests: ensure these folders are importable.
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'user_tools'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'user_tools', 'testRunner'))


class DummyConn:
    def __init__(self, responses=None):
        # One simple FIFO of responses consumed by sendData/getResult in call order.
        self._responses = list(responses or [])
        self.sent = []

    def sendData(self, dataJSON):
        self.sent.append(("POST", dataJSON))
        if self._responses:
            return self._responses.pop(0)
        return ""

    def getResult(self):
        self.sent.append(("GET", None))
        if self._responses:
            return self._responses.pop(0)
        return json.dumps({"alarmState": 0})


class TestDeviceAlgSettings(unittest.TestCase):
    def _make_alg(self, dummy_conn):
        import libosd.osdAppConnection

        with patch.object(libosd.osdAppConnection, "OsdAppConnection", lambda *args, **kwargs: dummy_conn):
            import user_tools.testRunner.deviceAlg as deviceAlg
            settings_str = json.dumps({"name": "device", "ipAddr": "127.0.0.1"})
            return deviceAlg.DeviceAlg(settings_str, debug=False)

    def test_get_settings_json_is_valid_json(self):
        dummy = DummyConn()
        alg = self._make_alg(dummy)

        obj = json.loads(alg.getSettingsJson())
        self.assertEqual(obj["dataType"], "settings")
        self.assertEqual(obj["analysisPeriod"], 5)
        self.assertEqual(obj["sampleFreq"], 25)
        self.assertEqual(obj["battery"], 0)
        self.assertEqual(obj["watchPartNo"], "n/a")
        self.assertEqual(obj["watchFwVersion"], "n/a")
        self.assertEqual(obj["sdVersion"], "n/a")
        self.assertEqual(obj["sdName"], "deviceAlg")

    def test_process_dp_replies_to_settings_request_from_post(self):
        # POST returns 'sendSettings', then after the settings handshake the GET returns analysis JSON.
        dummy = DummyConn(responses=[
            "sendSettings",  # response to first POST datapoint
            "",          # response to POST settings JSON
            "",          # response to re-POST datapoint
            json.dumps({"alarmState": 2}),  # response to GET /data
        ])
        alg = self._make_alg(dummy)

        ret = alg.processDp(json.dumps({"dataType": "raw", "data": []}), eventId="e1")
        obj = json.loads(ret)
        self.assertEqual(obj["alarmState"], 2)

        posted_payloads = [payload for method, payload in dummy.sent if method == "POST" and payload is not None]
        self.assertTrue(any(json.loads(p).get("dataType") == "settings" for p in posted_payloads))

    def test_process_dp_replies_to_settings_request_from_get(self):
        # POST ok, but GET returns 'sendSettings' once, then returns analysis.
        dummy = DummyConn(responses=[
            "",          # response to POST datapoint
            "sendSettings",  # response to GET /data
            "",          # response to POST settings JSON
            json.dumps({"alarmState": 1}),  # response to subsequent GET /data
        ])
        alg = self._make_alg(dummy)

        ret = alg.processDp(json.dumps({"dataType": "raw", "data": []}), eventId="e1")
        obj = json.loads(ret)
        self.assertEqual(obj["alarmState"], 1)

        posted_payloads = [payload for method, payload in dummy.sent if method == "POST" and payload is not None]
        self.assertTrue(any(json.loads(p).get("dataType") == "settings" for p in posted_payloads))


if __name__ == "__main__":
    unittest.main()
