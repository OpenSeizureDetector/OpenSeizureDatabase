#!/usr/bin/python3

import requests


class OsdAppConnection:
    ''' A class to manage a connection to an instance of the
    OpenSeizureDetector Android App web interface.
    '''
    def __init__(self, addr="192.168.1.29", port=8080, user='', passwd=''):
        self.addr = addr
        self.port = port
        self.user = user
        self.passwd = passwd

        print("OsdAppConnection - addr = %s" % addr)
        self.baseUrl = "http://%s:8080/" % addr
        print("OsdAppConnection - baseUrl = %s" % self.baseUrl)

    def _makeUrl(self, urlStr):
        """ Generate a full url based on the provided urlStr that defines
        the request path. """
        url = "http://%s:%d/%s" % ( self.addr, self.port, urlStr)
        return url

    def _sendRequest(self, urlStr, method="GET",
                     data=None, params=None, json=False):
        """ Send a http request to url urlStr using the specified method.
        params is encoded into the URL and data is sent with the request.
        if json is True it interprets the returned string as a json object,
        otherwise it returns the string itself. """
        #print("_sendRequest(%s, %s)" % (urlStr, method))
        url = self._makeUrl(urlStr)
        #print("url=%s" % url)
        if (method == "GET"):
            r = requests.get(url, auth=(self.user, self.passwd), params=params)
        elif (method == "POST"):
            r = requests.post(url, auth=(self.user, self.passwd), 
                              params=params,
                              data = data)
        else:
            print("Unsupported method %s" % method)
            return None
        if r.status_code == 200:
            if (json):
                return r.json()
            else:
                return r.text
        else:
            print("_get(%s): ERROR: status code = %d" % (urlStr,r.status_code))
            return None

    def _get(self, urlStr, params=None, json=False):
        """ send a GET request to url urlStr with params encoded into the URL.
        """
        # print("_get(%s)" % urlStr)
        return self._sendRequest(urlStr, "GET", params=params, json=json)

    def _post(self, urlStr, params=None, data=None, json=False):
        """send a POST request to url urlStr with params encoded into the URL
        and data sent with the request.
        """
        # print("_put(%s)" % urlStr)
        return self._sendRequest(urlStr, "POST",
                                 params=params, data=data, json=json)

    def sendData(self, dataJSON):
        """ Post the string dataJSON to the server.
        returns the response
        """
        urlStr = "data"
        #print("sendData: urlStr=%s, dataJSON=%s" % (urlStr,dataJSON))
        retVal = self._post(urlStr, None, dataJSON, False)
        #print(retVal)
        return retVal

    def getResult(self):
        """ Get the lastest analysis results from the device."""
        urlStr = "data"
        #print("getResult: urlStr=%s" % (urlStr))
        retVal = self._get(urlStr, None, False)
        #print(retVal)
        return retVal
        
    

if __name__=="_main__":
    print("osdAppConnection.main()")
