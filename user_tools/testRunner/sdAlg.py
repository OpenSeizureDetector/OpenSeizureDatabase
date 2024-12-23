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
        self.outFname = "algOutput-%s.csv" % self.settingsObj['name']
        self.outfile = open(self.outFname,'w')
        if self.outfile is None:
            print("ERROR opening output file %s" % self.outFname)

    def dateStr2secs(self, dateStr):
        parsed_t = dateutil.parser.parse(dateStr)
        return parsed_t.timestamp()


    def processDp(self, dpStr, eventId):
        self.logD("SdAlg.processDp: dpStr=%s." % dpStr)
        retVal = { "alarmState": 0 }
        return(json.dumps(retVal))
    
    def writeOutput(self, valArr):
        lineStr = ""
        first = True
        for val in valArr:
            if not first:
                lineStr = "%s," % lineStr
            first = False
            lineStr = "%s%s" % (lineStr,val)
        lineStr = "%s\n" % lineStr
        self.outfile.write(lineStr)


    def resetAlg(self):
        pass

    def close(self):
        self.outfile.close()


    def logD(self,msgStr):
        if (self.DEBUG):
            print(msgStr)

    def log(self, msgStr):
            print(msgStr)


if __name__ == "__main__":
    print("sdAlg.SdAlg.main()")
