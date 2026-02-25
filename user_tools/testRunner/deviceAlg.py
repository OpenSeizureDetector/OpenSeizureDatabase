#!usr/bin/python3

import json
import time
import sdAlg
import libosd
import libosd.osdAppConnection

class DeviceAlg(sdAlg.SdAlg):
    '''Implementation of a test algorithm that uses the network
    interface of a physical device to run the instance of OSD on the device
    and report the result.
    '''
    def __init__(self, settingsStr, debug=True):
        print("DeviceAlg.__init__() - settingsStr=%s" % settingsStr)
        print("DeviceAlg.__init__(): settingsStr=%s (%s)"
                               % (settingsStr, type(settingsStr)))
        super().__init__(settingsStr, debug)
        self.osdAppConnection = libosd.osdAppConnection.OsdAppConnection(
            self.settingsObj['ipAddr'])
        
        # Optional delay in milliseconds after each datapoint
        self.delayMs = self.settingsObj.get('delayMs', None)
        if self.delayMs is not None:
            print("DeviceAlg.__init__: Setting delay to %d ms after each datapoint" % self.delayMs)


    def getSettingsJson(self) -> str:
        """Return a settings JSON string compatible with the OSD server.

        This mirrors the GarminSD behaviour, but uses hard-coded values.
        """
        settings_obj = {
            "dataType": "settings",
            "analysisPeriod": 5,
            "sampleFreq": 25,
            "battery": 0,
            "watchPartNo": "n/a",
            "watchFwVersion": "n/a",
            "sdVersion": "n/a",
            "sdName": "deviceAlg",
        }
        return json.dumps(settings_obj)


    @staticmethod
    def _is_settings_request(resp) -> bool:
        # Server responds to a POST with either "OK" or "sendSettings".
        # Keep backward compatibility with older token "settings".
        if not isinstance(resp, str):
            return False
        token = resp.strip().lower()
        return token in {"sendsettings", "settings"}

        
    def processDp(self, dataJSON, eventId):
        #self.logD("DeviceAlg.processDp: dataJSON=%s." % dataJSON)
        #print(dataJSON)
        post_resp = self.osdAppConnection.sendData(dataJSON)

        # If the server requests settings, reply like GarminSD and
        # re-send the datapoint to avoid dropping a sample.
        if self._is_settings_request(post_resp):
            self.osdAppConnection.sendData(self.getSettingsJson())
            self.osdAppConnection.sendData(dataJSON)

        # Some server implementations request settings via the /data GET.
        # Ensure we always return valid JSON to the caller.
        retVal = None
        for _ in range(3):
            retVal = self.osdAppConnection.getResult()
            if self._is_settings_request(retVal):
                self.osdAppConnection.sendData(self.getSettingsJson())
                continue
            break

        # Fallback: never return a non-JSON string.
        if self._is_settings_request(retVal) or retVal is None:
            retVal = json.dumps({"valid": False, "alarmState": 0})
        
        # Apply delay if configured
        if self.delayMs is not None:
            time.sleep(self.delayMs / 1000.0)
        
        return(retVal)
                  
if __name__ == "__main__":
    print("deviceAlg.DeviceAlg.main()")
    settingsObj = {
        "ipAddr" : "192.168.1.162",
        }
    alg = DeviceAlg(json.dumps(settingsObj),True)
