#!/bin/sh

ncftpput -f ~/Dropbox/openseizuredetector.ftp /public_html/static/MLmodels \
	 MLmodels.json \
	 *.tflite

	 
