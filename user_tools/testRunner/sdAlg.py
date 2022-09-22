#!usr/bin/python3

import json
import dateutil.parser

class SdAlg:
    def __init__(self, settingsStr, debug=False):
        self.DEBUG = debug
        self.logD("SdAlg.__init__(): settingsStr=%s (%s)"
                               % (settingsStr, type(settingsStr)))
        self.settingsObj = json.loads(settingsStr)
        self.logD("SdAlg.__init__(): settingsObj="+json.dumps(self.settingsObj))

    def dateStr2secs(self, dateStr):
        parsed_t = dateutil.parser.parse(dateStr)
        return parsed_t.timestamp()


    def processDp(self, dpStr):
        self.logD("SdAlg.processDp: dpStr=%s." % dpStr)
        retVal = { "alarmState": 0 }
        return(json.dumps(retVal))

    def resetAlg(self):
        pass
    

    def logD(self,msgStr):
        if (self.DEBUG):
            print(msgStr)

    def log(self, msgStr):
            print(msgStr)


if __name__ == "__main__":
    print("sdAlg.SdAlg.main()")
