#!/usr/bin/env python

import os
import json
import requests
import dateutil.parser

def dateStr2secs(dateStr):
    parsed_t = dateutil.parser.parse(dateStr)
    return parsed_t.timestamp()


class WebApiConnection:
    DEBUG = False
    uname = "user"
    passwd = "user1_pw"
    baseUrl = "https://osdapi.ddns.net/api"
    cacheDir = os.path.join(os.path.expanduser("~"),"osd")
    cacheFname = "osd_data.json"
    download = True
    saveCache = True
    maxEvents = 10000

    def __init__(self, cfg=None, baseUrl=None, uname=None, passwd=None, cacheDir = None, download=True, saveCache=True, debug=False):
        self.download = download
        self.saveCache = saveCache
        self.DEBUG = debug
        if (self.DEBUG): print("libosd.WebApiConnection.__init__()")
        self.cfgFname = cfg
        if (cfg is not None):
            if (os.path.isfile(cfg)):
                print("Opening configuration file %s" % (cfg))
                with open(cfg) as infile:
                    jsonObj = json.load(infile)
                if (self.DEBUG): print(jsonObj)
                if ("uname" in jsonObj):
                    if (self.DEBUG): print("found uname - %s" % jsonObj["uname"])
                    self.uname = jsonObj["uname"]
                if ("passwd" in jsonObj):
                    if (self.DEBUG): print("found passwd - %s" % jsonObj["passwd"])
                    self.passwd = jsonObj["passwd"]
                if ("baseurl" in jsonObj):
                    if (self.DEBUG): print("found baseurl - %s" % jsonObj["baseurl"])
                    self.baseUrl = jsonObj["baseurl"]
                if ("cacheDir" in jsonObj):
                    if (self.DEBUG): print("found cacheDir - %s" % jsonObj["cacheDir"])
                    self.cacheDir = jsonObj["cacheDir"]
            else:
                print("ERROR - file %s does not exist" % cfg)
                exit(-1)
        else:
            print("No config file specified - using parameters")

        # Override config file parameters with command line parameters
        if (uname is not None):
            self.uname = uname
        if (passwd is not None):
            self.passwd = passwd
        if (baseUrl is not None):
            if (self.DEBUG): print("setting baseUrl")
            self.baseUrl = baseUrl
        if (cacheDir is not None):
            self.cacheDir = cacheDir

        if (self.DEBUG): print("baseUrl=%s, uname=%s, passwd=%s, cacheDir=%s" %
              (self.baseUrl, self.uname, self.passwd, self.cacheDir))

        if (self.download):
            print("webApiConnection - retrieving authentication token")
            self.getToken()
        else:
            print("webApiConnection - not downloading data so not logging in")
        
    def saveEventsCache(self,eventsLst):
        '''Write the list of events data eventsLst as a json file
        '''
        fpath = os.path.join(self.cacheDir, self.cacheFname)
        if (self.DEBUG): print("webApiConnection.saveEventsCache - fpath=%s" % fpath)
        fp = open(fpath,"w")
        fp.write(json.dumps(eventsLst, indent=2))
        fp.close()

    def loadEventsCache(self):
        ''' Retrieve a list of events data from a json file
        '''
        fpath = os.path.join(self.cacheDir, self.cacheFname)
        if (self.DEBUG): print("webApiConnection.loadEventsCache - fpath=%s" % fpath)
        fp = open(fpath,"r")
        eventsLst = json.load(fp)
        fp.close()
        return eventsLst

            
    def getEvents(self, userId=None, includeDatapoints = False):
        ''' Returns a list of all events in the database for the given userId
        or for all users if userId is None.
        The returned data does NOT contain the datapoints associated 
        with the events unless includeDatapoints is set to True.
        FIXME - add data range options
        '''
        if (self.DEBUG): print("libOsd.getEvents, baseUrl=%s" % (self.baseUrl))
        # If we are not downloading data, just return what we have cached
        if (not self.download):
            return self.loadEventsCache()

        # Otherwise download the specified data
        if (userId is not None):
            urlStr = "%s/events/?user=%s" % (self.baseUrl, userId)
        else:
            urlStr = "%s/events/" % (self.baseUrl)
        if (self.DEBUG): print("getEvents - urlStr=%s" % urlStr)
        eventsObj = self.getData(urlStr,None)

        ###############################################################
        # Return just the events list, unless includeDatapoints is True.
        if includeDatapoints == False:
            print("includeDatapoints is False - returning list of events without datapoints")
            # Cache the data in case we need it next time
            if (self.saveCache):
                self.saveEventsCache(eventsObj)
            return eventsObj

        eventLst = []
        count = 0
        for event in eventsObj:
            print("%5d %s %s %s %s" % (event['id'],
                                       event['dataTime'],
                                       event['type'],
                                       event['subType'],
                                       event['desc']
                                       ))
            dataPointsObj = self.getDataPointsByEvent(event['id'])
            # Make sure we are sorted into time order
            dataPointsObj.sort(key=lambda dp: dateStr2secs(dp['dataTime']))

            # Extract data from first datapoint to get OSD settings
            #     at time of event.
            if len(dataPointsObj)!=0:
                event['datapoints'] = dataPointsObj
                eventLst.append(event)
            else:
                print("Ignoring event with zero datapoints")

            count = count + 1
            if count >= self.maxEvents:
                print("reached maxEvents (%d) - stopping" % self.maxEvents)
                break

        # Cache the data in case we need it next time
        if (self.saveCache):
            self.saveEventsCache(eventLst)
        return eventLst


        
        
    
    def getUnvalidatedEvents(self, userId =1):
        if (self.DEBUG): print("libOsd.getUnvalidatedEvents, userId=%d, baseUrl=%s" % (userId, self.baseUrl))
        if (userId is not None):
            urlStr = "%s/events/?type__isnull=true&user=%d" % (self.baseUrl, userId)
        else:
            urlStr = "%s/events/?type__isnull=true" % (self.baseUrl)
        if (self.DEBUG): print("getUnvalidatedEvents - urlStr=%s" % urlStr)
        retVal = self.getData(urlStr,None)
        return retVal

    def getEvent(self, eventId, includeDatapoints=False):
        # If we are not downloading data, just return what we have cached
        if (not self.download):
            eventsLst = self.loadEventsCache()
            for event in eventsLst:
                if (event['id']==eventId):
                    return event
            print("Event not found in cache")
            return None

        if (self.DEBUG): print("libOsd.getEvent, eventId=%d, baseUrl=%s" % (eventId, self.baseUrl))
        urlStr = "%s/events/%d" % (self.baseUrl, eventId)
        if (self.DEBUG): print("getEvent - urlStr=%s" % urlStr)
        eventObj = self.getData(urlStr,None)
        
        ###############################################################
        # Return just the event list, unless includeDatapoints is True.
        if includeDatapoints:
            dataPointsObj = self.getDataPointsByEvent(eventObj['id'])
            # Make sure we are sorted into time order
            dataPointsObj.sort(key=lambda dp: dateStr2secs(dp['dataTime']))
            if len(dataPointsObj)!=0:
                eventObj['datapoints'] = dataPointsObj
        return eventObj

    def addEvent(self, eventType, dataTime, desc, wearerId):
        data = {
            "eventType": eventType,
            "dataTime": dataTime,
            "desc" : desc,
            "userId": wearerId
            }
        urlStr = "%s/events/" % self.baseUrl
        if (self.DEBUG): print("addEvent - urlStr=%s" % urlStr)
        retVal = self.postData(urlStr,data)
        if (self.DEBUG): print("addEvent - retVal=",retVal)
        return retVal

    def updateEvent(self, eventId, eventType = None, dataTime = None, desc = None, wearerId = None):
        data = self.getEvent(eventId)
        if (self.DEBUG): print("updateEvent - eventId=%d, data=" % eventId,data)
        if (eventType is not None):
            data['eventType'] = eventType
        if (dataTime is not None):
            data['dataTime'] = dataTime
        if (desc is not None):
            data['desc'] = desc
        if (wearerId is not None):
            data['wearerId'] = wearerId
        urlStr = "%s/events/%d/" % (self.baseUrl, eventId)
        if (self.DEBUG): print("updateEvent - urlStr=%s" % urlStr)
        retVal = self.putData(urlStr,data)
        return retVal
        
    def getDataPointsByEvent(self, eventId =1):
        if (self.DEBUG): print("libOsd.getDataPointsByEvent, eventId=%d, baseUrl=%s" % (eventId, self.baseUrl))
        urlStr = "%s/datapoints/?eventId=%d" % (self.baseUrl, eventId)
        if (self.DEBUG): print("getDataPointsByEvent - urlStr=%s" % urlStr)
        retVal = self.getData(urlStr,None)
        return retVal

    def getUser(self, userId=None):
        if (self.DEBUG):
            print("libosd.webApiConnection.getUser(): userId=%s, baseUrl=%s" % (userId, self.baseUrl))
        if (userId is None):
            urlStr = "%s/accounts/profile/" % (self.baseUrl)
        else:
            urlStr = "%s/accounts/profile/%d" % (self.baseUrl, userId)
        if (self.DEBUG): print("getUser - urlStr=%s" % urlStr)
        retVal = self.getData(urlStr,None)
        if (self.DEBUG): print("getUser, returning: ",retVal)
        return retVal
 
    def uploadFile(self, fname, wearerId=1):
        print("libosd.uploadFile")
        #urlStr = "%s/datapoints/add.json" % self.baseUrl
        #urlStr = "%s/datapoints/" % self.baseUrl
        urlStr = "%s/uploadCsvData/" % self.baseUrl
        print("libosd: urlStr=%s" % urlStr)
        if (os.path.isfile(fname)):
            print("Opening file %s" % (fname))
            with open(fname) as infile:
                lineStr = "start"
                lineCount = 0
                #while (lineStr):
                #    lineStr = infile.readline()
                    # print(lineStr)
                #    self.postData(urlStr,
                #                  lineStr)
                    #lineStr = None
                lineStrs = infile.readlines()
                print("%d lines read from file" % len(lineStrs))
                self.postData(urlStr, lineStrs)
                print("libosd.uploadFile() - eof - linecount=%d" % lineCount)

    def addDatapoint(self, eventId, dataTime, wearerId):
        data = {
            "eventId": eventId,
            "dataTime": dataTime,
            "userId": wearerId
            }
        urlStr = "%s/datapoints/" % self.baseUrl
        if (self.DEBUG): print("addDatapoint - urlStr=%s" % urlStr)
        retVal = self.postData(urlStr,data)
        if (self.DEBUG): print("addDatapoint - retVal=",retVal)
        return retVal


    def postData(self, url, data, toObj=True):
        headerObj = {
                "Authorization": "Token %s" % self.token
        }
        if (self.DEBUG): print("libosd.postData() - url=%s, data=%s" % (url, data))
        if (self.DEBUG): print("libosd.postData() - headerObj=",headerObj)
        response = requests.post(
            url,
            headers=headerObj,
            #auth=(self.uname, self.passwd),
            json=data)
        # print("postData() - response=%s" % response.text)
        # print(response.status)
        # print(response.reason)
        if (self.DEBUG): print("libosd.postdata(): Status Code=%d" % response.status_code)
        # print(dir(response))
        if (self.DEBUG): print("libosd.postdata(): Response=", response.text)
        if (toObj):
            retVal = json.loads(response.text)
        else:
            retVal = response.txt
        if (self.DEBUG): print("libosd.postdata(): Returning=", retVal)
        return(retVal)


    def putData(self, url, data):
        headerObj = {
                "Authorization": "Token %s" % self.token
        }
        if (self.DEBUG): print("libosd.postData() - url=%s, data=%s" % (url, data))
        if (self.DEBUG): print("libosd.postData() - headerObj=",headerObj)
        response = requests.put(
            url,
            headers=headerObj,
            #auth=(self.uname, self.passwd),
            json=data)
        # print("postData() - response=%s" % response.text)
        # print(response.status)
        # print(response.reason)
        if (self.DEBUG): print("libosd.putdata(): Status Code=%d" % response.status_code)
        # print(dir(response))
        if (self.DEBUG): print("libosd.putdata(): Response=", response.text)
        return(response.text)


    def getData(self, url, data,toObj=True):
        headerObj = {
                "Authorization": "Token %s" % self.token
        }
        if (self.DEBUG): print("libosd.getData() - url=%s, data=%s" % (url, data))
        if (self.DEBUG): print("libosd.getData() - headerObj=",headerObj)
        response = requests.get(
            url,
            headers=headerObj,
            #auth=(self.uname, self.passwd),
            json=data)
        # print("getData() - response=%s" % response.text)
        if (self.DEBUG): print("libosd.getdata(): Status Code=%d" % response.status_code)
        if (response.status_code==200):
            # print(dir(response))
            if (toObj):
                retVal = json.loads(response.text)
            else:
                retVal = response.txt
        else:
            retVal = None
        if (self.DEBUG): print("libosd.getdata(): Returning") #=", retVal)
        return(retVal)

    def getToken(self):
        urlStr = "%s/accounts/login/" % self.baseUrl
        if (self.DEBUG): print("webApiConnection.getToken(): urlStr=%s" % urlStr)
        if (self.DEBUG): print("webApiConnection.getToken(): %s, %s" %
                               (self.uname, self.passwd))
        response = requests.post(
            urlStr,
            json = {
                "login": self.uname,
		"password": self.passwd
                }
        )
        if (self.DEBUG): print("Status Code=%d" % response.status_code)
        if (response.status_code == 200):
            jsonObj = json.loads(response.text)
            self.token = jsonObj['token']
            if (self.DEBUG): print("token=%s" % self.token)
        else:
            self.token = None
            print("ERROR - Token not set")
        # print(dir(response))
        if (self.DEBUG): print("libosd.getToken(): Response Headers", response.headers)
        if (self.DEBUG): print("libosd.getToken(): response.txt=",response.text)
        return(response.text)




        
        
if (__name__ == "__main__"):
    print("libosd.main()")
    osd = WebApiConnection(cfg="client.cfg", uname="graham4", passwd="testpwd1", debug=True)
    #osd.uploadFile("DataLog_2019-11-04.txt", wearerId=3)
    #eventsObj = osd.getEvents()
    #print("eventsObj = ", eventsObj)
    usersObj = osd.getUser()
    print("usersObj = ", usersObj)
    usersObj = osd.getUser(39)
    print("usersObj = ", usersObj)
    #unvalidatedEventsObj = osd.getUnvalidatedEvents()
    #print("unvalidatedEventsObj = ", unvalidatedEventsObj)

    #retVal = osd.addEvent(eventType=4, dataTime="2021-11-30T20:35:00Z",
    #                      desc="testing addEvent",
    #                      wearerId=3)
    #print("addEvent - retVal=",retVal)
    #print("addEvent - new EventId = %d" % retVal['id'])

    #eventsObj = osd.getEvents()
    #print("eventsObj = ", eventsObj)
    #print(eventsObj['results'])

    #retVal = osd.updateEvent(eventId=2, eventType=3)
    #print(retVal)


    #retVal = osd.addDatapoint(eventId=4, dataTime="2021-11-30T20:35:00Z",
    #                          wearerId=4)
    #print("addDatapoint - retVal=",retVal)
    
