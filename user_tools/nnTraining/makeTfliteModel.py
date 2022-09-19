#!/usr/bin/env python

import argparse
import os
import tensorflow as tf




if (__name__ == "__main__"):
    print("makeTfliteModel.main")
    parser = argparse.ArgumentParser(description='Convert a keras model to tensorflow Lite format')
    parser.add_argument('inFile',
                        help='input filename (Without h5 extension)')
    parser.add_argument('--debug', action="store_true",
                        help='Write debugging information to screen')
    argsNamespace = parser.parse_args()
    args = vars(argsNamespace)
    print(args)

    filepath = os.path.basename(args['inFile'])
    print("filepath=%s" % filepath)
    fileroot = os.path.splitext(filepath)[0]
    print("fileroot=%s" % fileroot)

    inFname = "%s.h5" % fileroot
    outFname = "%s.tflite" % fileroot

    print("inFname=%s" % inFname)
    print("outFname=%s" % outFname)

    model = tf.keras.models.load_model(inFname)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(outFname, 'wb') as f:
        f.write(tflite_model)

    
