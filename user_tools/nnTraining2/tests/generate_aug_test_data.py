#!/usr/bin/env python3
import argparse

nAcc = 125  # Number of accelerometer datapoints per datapoint

def generate_header(outFile):
    outFile.write("eventId,type,userId")

    for i in range(nAcc):
        outFile.write(",")
        outFile.write(f"M{i:03d}")
    for i in range(nAcc):
        outFile.write(",")
        outFile.write(f"X{i:03d}")
    for i in range(nAcc):
        outFile.write(",")
        outFile.write(f"Y{i:03d}")
    for i in range(nAcc):
        outFile.write(",")
        outFile.write(f"Z{i:03d}")

def generate_event(eventId, nDp, eventType, outFile, debug=False):
    accVal = 1000.0*eventId
    xVal = 1001.0*eventId
    yVal = 1002.0*eventId
    zVal = 1003.0*eventId

    for dpId in range(nDp):

        outFile.write(f"\n{eventId},{eventType},123")

        for i in range(nAcc):
            outFile.write(f",{accVal:8.3f}")
            accVal += 0.001
        for i in range(nAcc):
            outFile.write(f",{xVal:8.3f}")
            xVal += 0.001
        for i in range(nAcc):
            outFile.write(f",{yVal:8.3f}")
            yVal += 0.001
        for i in range(nAcc):
            outFile.write(f",{zVal:8.3f}")
            zVal += 0.001

        if debug:
            print(f"Generated eventId {eventId} datapoint {dpId} of type {eventType}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate test data for testing nnTraining2 augmentation functions"
    )
    parser.add_argument(
        "outFile", type=str, help="Output Filename"
    )
    parser.add_argument(
        "--nEvents",
        type=int,
        default=3,
        help="Number of events to generate",
    )
    parser.add_argument(
        "--nDp",
        type=int,
        default=5,
        help="Number of datapoints per event to generate (generates 50% seizure, 50% non-seizure events)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output",
    )

    args = vars(parser.parse_args())

    print(args)

    outFile = open(args['outFile'], 'w')

    generate_header(outFile)

    for eventId in range(args['nEvents']):
        if eventId % 2 == 0:
            eventType = 1  # seizure
        else:
            eventType = 0  # non-seizure
        generate_event(eventId+1, args['nDp'], eventType, outFile, args['debug'])

    outFile.close()

    print(f"Generated {args['outFile']} with {args['nEvents']} events of {args['nDp']} datapoints each" )