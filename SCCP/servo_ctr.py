import os.path
import re
import sys
import cv2
import time
import datetime
import glob
import RPi.GPIO as GPIO
import numpy as np
import tflite_runtime.interpreter as tflite
from six.moves import urllib
from PIL import Image, ImageFilter, ImageChops
from operator import itemgetter

input_mean = 127.5
input_std = 127.5

modelfile = "converted_model.tflite"

classname = ["left", "right", "straight"]

#dir
imagedir = "image/"

def getPicture(currenttime, cap):
	jpegfile = imagedir + currenttime + '.jpg'
	_, img = cap.read()
	cv2.imwrite(jpegfile, img)
	return jpegfile
	
def set_input_tensor(interpreter, image):
	tensor_index = interpreter.get_input_details()[0]['index']
	input_tensor = interpreter.tensor(tensor_index)()[0]
	input_tensor[:, :] = image
	
def classify_image(interpreter, image):
	input_details = interpreter.get_input_details()
	floating_model = input_details[0]['dtype'] == np.float32
	print("floating", floating_model)
	input_data = np.expand_dims(image, axis=0)
	if floating_model:
		input_data = (np.float32(input_data) - input_mean) / input_std
		
	set_input_tensor(interpreter, input_data)
	interpreter.invoke()
	output_details = interpreter.get_output_details()[0]
	output = np.squeeze(interpreter.get_tensor(output_details['index']))
	
	if output_details['dtype'] == np.uint8:
		scale, zero_point = output_details['quantization']
		output = scale * (output - zero_point)
	
	print("prediction result")
	print(output)
	return output
	
def CheckPredictionResult(predictions):
	maxindex = np.argmax(predictions)
	direction = classname[maxindex]
	return direction

def main():
	cap = cv2.VideoCapture(0)
	cap.set(cv2.CAP_PROP_FPS, 10)
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
	
	interpreter = tflite.Interpreter(modelfile)
	interpreter.allocate_tensors()
	_, height, width, _ = interpreter.get_input_details()[0]['shape']
	print(height, width)
	
	#open JPG file
	#jpegfile = getPicture("aaa", cap)
	jpegfile = "test/img_20190808_030047_100005.jpg"
	image_data = Image.open(jpegfile).resize((width, height))
	#inference
	predictions = classify_image(interpreter, image_data)
	direction = CheckPredictionResult(predictions)
	
	print(direction)
	
	servo = 2

	GPIO.setmode(GPIO.BCM)
	GPIO.setwarnings(False)
	GPIO.setup(servo, GPIO.OUT, initial=GPIO.LOW)
	p = GPIO.PWM(servo, 50)
	p.start(0)
	p.ChangeDutyCycle(7.5)
	time.sleep(1)
	if direction == 'left':
		p.ChangeDutyCycle(2.5)
		time.sleep(1)
	elif direction == 'right':
		p.ChangeDutyCycle(12)
		time.sleep(1)
	else:
		p.ChangeDutyCycle(7.5)
		time.sleep(1)
	
	GPIO.cleanup()
	cap.release()

main()

