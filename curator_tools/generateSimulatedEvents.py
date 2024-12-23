#!/usr/bin/env python

from datetime import datetime
import dateutil
import math
import json
import matplotlib.pyplot as plt


def dateStr2secs(dateStr):
    parsed_t = dateutil.parser.parse(dateStr)
    return parsed_t.timestamp()

def secs2dateStr(timestamp):
    date_time = datetime.fromtimestamp(timestamp)
    # "dataTime": "2023-05-05T06:28:47Z"
    dStr = date_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    return(dStr)


def generateSimulatedEvent(eventSpec):
    samplePeriod = 5.0    # Sample period in seconds.
    print("generateSimulatedEvent()")
    desc = eventSpec['desc']
    startTimeSecs = dateStr2secs(eventSpec['startDate'])
    sampleFreq = eventSpec['sampleFreq']
    durationSecs = eventSpec['durationSecs']
    componentsLst = eventSpec['components']


    nSamp = int(durationSecs * sampleFreq)
    nDpSamp = int(samplePeriod * sampleFreq)
    nDp = int(durationSecs / samplePeriod)

    print("desc=%s" % desc)
    print("sampleFreq=%.1f, duration=%.1f, nSamp=%d, nDpSamp=%d, nDp=%d" % (sampleFreq, durationSecs, nSamp, nDpSamp, nDp))
    print("components=", componentsLst)

    timeLst = []
    magLst = []
    xLst = []
    yLst = []
    zLst = []


    eventObj = {}
    dataPointsLst = []
    dp3dLst = []   # Acceleration cartesian components x1, y1, z1, x2, y2, z2, ....xn, yn, zn in milli-g
    dpRawLst = []  # Acceleration vector magnitudes in milli-g

    # Step through the entire event one measurement at a time.
    for n in range(0,nSamp):
        # Time from the start of the event.
        timeSecs = n / sampleFreq
        print("timeSecs=%.2f" % timeSecs)

        # Initialise each component of acceleration as zero
        x = 0.0
        y = 0.0
        z = 0.0

        # Then add on the contribution from each component specified in the event specification object.
        for c in componentsLst:
            if (timeSecs >= c['startSecs']) and (timeSecs < c['endSecs']):
                print("Component %s active" % c['desc'])
                a = c['ampl']*math.cos(2*math.pi * c['freq']*timeSecs + c['phase']*2*math.pi/360.)
                if c['axis'] == 0:
                    x = x + a
                elif c['axis'] == 1:
                    y = y + a
                elif c['axis'] == 2:
                    z = z + a
                else:
                    print("***ERROR*** - Invalid axis %s" % c['axis'])

            pass

        # Append the acceleration values to the magnitude and 3d acceleration lists.
        mag = math.sqrt(x*x + y*y + z*z)
        dpRawLst.append(mag)
        dp3dLst.append(x)
        dp3dLst.append(y)
        dp3dLst.append(z)

        # Append the values to the whole event lists that are used for debugging
        timeLst.append(timeSecs)
        magLst.append(mag)
        xLst.append(x)
        yLst.append(y)
        zLst.append(z)

        # If we have got to the end of a 5 second sample period, save the data as a 'datapoint' (sub-event) object.
        if len(dpRawLst) == nDpSamp:
            dpObj = {}
            dataTimeSecs = startTimeSecs+timeSecs
            dpObj['dataTime'] = secs2dateStr(dataTimeSecs)
            dpObj['eventId'] = eventSpec['id']
            dpObj['rawData'] = dpRawLst
            dpObj['rawData3D'] = dp3dLst
            dpObj['hr'] = 0.0
            dpObj['o2sat'] = 0.0
            dpObj['simpleSpec'] = [0,0,0,0,0,0,0,0,0,0]

            dataPointsLst.append(dpObj)
            dpRawLst = []
            dp3dLst = []

    eventObj['type'] = 'simulation'
    eventObj['subType'] = 'calculated'
    eventObj['desc'] = eventSpec['desc']
    eventObj['id'] = eventSpec['id']
    eventObj['userId'] = '1'
    eventObj['osdAlarmState'] = 0
    eventObj['sampleFreq'] = eventSpec['sampleFreq']
    eventObj['datapoints'] = dataPointsLst
    eventObj['dataTime'] = secs2dateStr(startTimeSecs)
    eventObj['dataSourceName'] = 'generatedSimulatedEvents.py'

    # Plot the event data to check it looks correct.
    fig, ax = plt.subplots(4,1)
    ax[0].plot(timeLst, magLst)
    ampl = max(magLst)-min(magLst)
    print(magLst)
    print("amplitude=%.3f" % ampl)
    #ax[0].set_ylim([900,1000+ampl])
    ax[1].plot(timeLst, xLst)
    ax[2].plot(timeLst, yLst)
    ax[3].plot(timeLst, zLst)
    plotFname = "plot.png"
    fig.savefig(plotFname)
    print("Acceleration Plot saved to %s" % plotFname)

    return eventObj


if __name__ == "__main__":
    print("generateSimulatedEvents.py - main()")

    eventSpecs = [
        {
            'desc': 'Simulated - gravity (x axis) only',
            'id': 'S001',
            'startDate': "1983-05-01T00:00:00Z",
            'sampleFreq': 25.0,
            'durationSecs': 180.0,
            'components': [
                {
                    'desc': 'Gravity - x axis',
                    'axis': 0,
                    'startSecs': 0.0,
                    'endSecs': 180.0,
                    'freq': 0.0,
                    'phase': 0.0,
                    'ampl': 1000.0
                }
            ]
        },
        {
            'desc': 'Simulated - 5Hz (y), gravity (x)',
            'id': 'S002',
            'startDate': "1983-05-01T00:00:00Z",
            'sampleFreq': 25.0,
            'durationSecs': 180.0,
            'components': [
                {
                    'desc': 'Gravity - x axis',
                    'axis': 0,
                    'startSecs': 0.0,
                    'endSecs': 180.0,
                    'freq': 0.0,
                    'phase': 0.0,
                    'ampl': 1000.0
                },
                {
                    'desc': '5Hz - y axis',
                    'axis': 1,
                    'startSecs': 60.0,
                    'endSecs': 90.0,
                    'freq': 5,
                    'phase': 270.0,
                    'ampl': 300
                },
            ]
        },
        {
            'desc': 'Simulated - 3Hz (y), gravity (x)',
            'id': 'S003',
            'startDate': "1983-05-01T00:00:00Z",
            'sampleFreq': 25.0,
            'durationSecs': 180.0,
            'components': [
                {
                    'desc': 'Gravity - x axis',
                    'axis': 0,
                    'startSecs': 0.0,
                    'endSecs': 180.0,
                    'freq': 0.0,
                    'phase': 0.0,
                    'ampl': 1000.0
                },
                {
                    'desc': '3Hz - y axis',
                    'axis': 1,
                    'startSecs': 60.0,
                    'endSecs': 90.0,
                    'freq': 3,
                    'phase': 270.0,
                    'ampl': 300
                }
            ]
        },
        {
            'desc': 'Simulated - 1Hz (y), 3Hz (z), 3 Hz (x), gravity (x)',
            'id': 'S004',
            'startDate': "1983-05-01T00:00:00Z",
            'sampleFreq': 25.0,
            'durationSecs': 180.0,
            'components': [
                {
                    'desc': 'Gravity - x axis',
                    'axis': 0,
                    'startSecs': 0.0,
                    'endSecs': 180.0,
                    'freq': 0.0,
                    'phase': 0.0,
                    'ampl': 1000.0
                },
                {
                    'desc': '1Hz - y axis',
                    'axis': 1,
                    'startSecs': 10.0,
                    'endSecs': 60.0,
                    'freq': 1,
                    'phase': 270.0,
                    'ampl': 400
                },
                {
                    'desc': '3Hz - z axis',
                    'axis': 2,
                    'startSecs': 60.0,
                    'endSecs': 90.0,
                    'freq': 3,
                    'phase': 180.0,
                    'ampl': 600
                },
                {
                    'desc': '3Hz - x axis',
                    'axis': 0,
                    'startSecs': 90.0,
                    'endSecs': 120.0,
                    'freq': 3,
                    'phase': 90.0,
                    'ampl': 100
                }

            ]
        }

    ]
    

    componentTemplate = {
                'desc': 'Gravity - x axis',
                'axis': 0,
                'startSecs': 0.0,
                'endSecs': 180.0,
                'freq': 0.0,
                'phase': 0.0,
                'ampl': 1000.0
            }


    eventsLst=[]
    for eventSpec in eventSpecs:
        print("eventSpec=",eventSpec)
        eventObj = generateSimulatedEvent(eventSpec)

        #print(eventObj)

        for dp in eventObj['datapoints']:
            print("len(rawData)=%d, len(rawData3D)=%d" % (len(dp['rawData']), len(dp['rawData3D'])))

        eventsLst.append(eventObj)

    fname = "simulated_events.json"
    with open(fname,"w") as fp:
        json.dump(eventsLst, fp)
    print("Events data written to file %s" % fname)