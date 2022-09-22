#!usr/bin/python3

import json
import sdAlg
import libosd

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

        
    def processDp(self, dataJSON):
        #self.logD("DeviceAlg.processDp: dataJSON=%s." % dataJSON)
        #print(dataJSON)
        retVal = self.osdAppConnection.sendData(dataJSON)
        #print("retVal=",retVal)
        retVal = self.osdAppConnection.getResult()
        #print("retVal=",retVal)
        return(retVal)
                  
if __name__ == "__main__":
    print("deviceAlg.DeviceAlg.main()")
    settingsObj = {
        "ipAddr" : "192.168.1.162",
        }
    alg = DeviceAlg(json.dumps(settingsObj),True)
