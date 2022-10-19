import webiopi
import os
import sys
import cv2
import time
import datetime
import glob
import RPi.GPIO as GPIO
import numpy as np
#import tflite_runtime.interpreter as tflite
from six.moves import urllib
from PIL import Image, ImageFilter, ImageChops
from operator import itemgetter

input_mean = 127.5
input_std = 127.5

modelfile = "converted_model.tflite"
#modelfile = "quantized_model.tflite"

classname = ["left", "right", "straight"]
LEFT = 21
RIGHT = 20
#dir
imagedir = "/home/pi/tank/desk_data/"

def getPicture(nowtime, direction):
    jpegfile = imagedir + direction + '/' + nowtime.replace(' ', '-') + '.jpg'
    cmd = 'wget --http-user="AAA" --http-password="BBB"  -O ' + jpegfile + r' http://192.168.137.110:8080/?action=snapshot'
    #print(cmd)
    os.system(cmd)
@webiopi.macro
def cap(no):
	GPIO.setmode(GPIO.BCM)
	GPIO.setwarnings(False)
	GPIO.setup(LEFT, GPIO.OUT, initial=GPIO.LOW)
	GPIO.setup(RIGHT, GPIO.OUT, initial=GPIO.LOW)

	if no == 'l':
		GPIO.output(LEFT, 1)
		GPIO.output(RIGHT, 0)
	elif no == 'r':
		GPIO.output(LEFT, 0)
		GPIO.output(RIGHT, 1)
	elif no == 's':
		GPIO.output(LEFT, 1)
		GPIO.output(RIGHT, 1)
	else:
		GPIO.output(LEFT, 1)
		GPIO.output(RIGHT, 1)
	time.sleep(0.1)
	GPIO.output(LEFT, 0)
	GPIO.output(RIGHT, 0)
	nowtime = datetime.datetime.now()
	getPicture(str(nowtime), no)
	GPIO.cleanup()	

#cap('r')