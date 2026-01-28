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

        
    def processDp(self, dataJSON, eventId):
        #self.logD("DeviceAlg.processDp: dataJSON=%s." % dataJSON)
        #print(dataJSON)
        retVal = self.osdAppConnection.sendData(dataJSON)
        #print("retVal=",retVal)
        retVal = self.osdAppConnection.getResult()
        #print("retVal=",retVal)
        
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
