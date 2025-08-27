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
            dpObj['id'] = f"{eventSpec['id']}_{len(dataPointsLst):03d}"
            dpObj['rawData'] = dpRawLst
            dpObj['rawData3D'] = dp3dLst
            dpObj['hr'] = 0.0
            dpObj['o2sat'] = 0.0
            dpObj['simpleSpec'] = [0,0,0,0,0,0,0,0,0,0]

            dataPointsLst.append(dpObj)
            dpRawLst = []
            dp3dLst = []

    eventObj['type'] = eventSpec.get('type', 'nda')
    eventObj['subType'] = 'simulation'
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


    # Add three 'test' events for runSequence testing
    test_event_duration = 25.0  # seconds
    test_sample_freq = 25.0
    test_sample_period = 5.0
    test_points = int(test_event_duration // test_sample_period)
    test_events = []
    for axis in range(3):
        event_id = f"T00{axis+1}"
        eventSpec = {
            'desc': f'Test event - axis {axis} sequence',
            'id': event_id,
            'startDate': "2025-08-24T00:00:00Z",
            'sampleFreq': test_sample_freq,
            'durationSecs': test_event_duration,
            'components': []
        }
        dataPointsLst = []
        for dp_idx in range(test_points):
            dpObj = {}
            # Each datapoint covers 5 seconds, 125 samples
            dpObj['dataTime'] = secs2dateStr(dateStr2secs(eventSpec['startDate']) + dp_idx * test_sample_period)
            dpObj['eventId'] = event_id
            dpObj['id'] = f"{event_id}_{dp_idx:03d}"
            rawData3D = []
            rawData = []
            for i in range(int(test_sample_freq * test_sample_period)):
                val = dp_idx * int(test_sample_freq * test_sample_period) + i
                arr = [0, 0, 0]
                arr[axis] = val
                rawData3D.extend(arr)
                rawData.append(abs(val))
            dpObj['rawData3D'] = rawData3D
            dpObj['rawData'] = rawData
            dpObj['hr'] = 0.0
            dpObj['o2sat'] = 0.0
            dpObj['simpleSpec'] = [0,0,0,0,0,0,0,0,0,0]
            dataPointsLst.append(dpObj)
        eventObj = {
            'type': 'test',
            'subType': 'sequence',
            'desc': eventSpec['desc'],
            'id': event_id,
            'userId': 'test',
            'osdAlarmState': 0,
            'sampleFreq': test_sample_freq,
            'datapoints': dataPointsLst,
            'dataTime': eventSpec['startDate'],
            'dataSourceName': 'generateSimulatedEvents.py'
        }
        test_events.append(eventObj)

    print("generateSimulatedEvents.py - main()")

    eventSpecs = [
        {
            'desc': 'Simulated - gravity (x axis) only',
            'id': 'S001',
            'type': 'seizure',
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
            'type': 'seizure',
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
            'type': 'nda',
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
            'type': 'nda',
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
        for dp in eventObj['datapoints']:
            print("len(rawData)=%d, len(rawData3D)=%d" % (len(dp['rawData']), len(dp['rawData3D'])))
        eventsLst.append(eventObj)

    # Add the test events
    eventsLst.extend(test_events)

    fname = "simulated_events.json"
    import json
    def compact_arrays(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in ["rawData", "rawData3D"] and isinstance(v, list):
                    # Convert array to compact string
                    obj[k] = json.dumps(v, separators=(",", ":"))
                else:
                    compact_arrays(v)
        elif isinstance(obj, list):
            for item in obj:
                compact_arrays(item)
    # Make a deep copy to avoid modifying in-memory data
    import copy
    compact_events = copy.deepcopy(eventsLst)
    compact_arrays(compact_events)
    # Dump with indent, then replace compact array strings with actual arrays
    json_str = json.dumps(compact_events, indent=2)
    import re
    # Replace quoted array strings with actual arrays
    json_str = re.sub(r'"(\[.*?\])"', lambda m: m.group(1), json_str)
    with open(fname, "w") as fp:
        fp.write(json_str)
    print("Events data written to file %s" % fname)