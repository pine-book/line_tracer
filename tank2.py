import os.path
import re
import sys
import cv2
import time
import datetime
import glob
import RPi.GPIO as GPIO
import numpy as np
import time
import tflite_runtime.interpreter as tflite
from PIL import Image

input_mean = 127.5
input_std = 127.5

modelfile = "converted_model.tflite"
LEFT = 21
RIGHT = 20

classname = ["l","r"]

#dir
imagedir = "image/"

def getPicture(currenttime, cap):
    jpegfile = imagedir + currenttime + '.jpg'
    _, img = cap.read()
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _, img_dst = cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    img_data = img_dst

    cv2.imwrite(jpegfile, img_data)
    return jpegfile

def set_input_tensor(interpreter, image):
	tensor_index = interpreter.get_input_details()[0]['index']
	input_tensor = interpreter.tensor(tensor_index)()[0]
	#input_tensor[:, :] = image
	input_tensor[:, :] = (224,224,3)

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

    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup(LEFT, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(RIGHT, GPIO.OUT, initial=GPIO.LOW)
    interpreter = tflite.Interpreter(modelfile)
    interpreter.allocate_tensors()
    _, height, width, dim = interpreter.get_input_details()[0]['shape']
    print(height, width, dim)
    print(interpreter.get_input_details()[0]['shape'])
    try:
        while True:
            nowtime = datetime.datetime.now()
            jpegfile = getPicture(str(nowtime), cap)
            #image_data = Image.open(jpegfile).resize((width, height))
            input_details = interpreter.get_input_details()
            #print(input_details)
            input_shape = input_details[0]['shape']
            tmp = cv2.imread(jpegfile)
            tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
            tmp = cv2.resize(tmp, (224, 224))
            #input_data = np.expand_dims(image_data, axis=0).astype("float32")
            input_data = np.expand_dims(tmp, axis=0).astype("float32")
            interpreter.set_tensor(input_details[0]['index'], input_data)

            interpreter.invoke()
            output_details = interpreter.get_output_details()
            output_data =interpreter.get_tensor(output_details[0]['index'])
            print(output_data)
            maxindex = np.argmax(output_data)
            direction = classname[maxindex]
            print(direction)
            if direction == 'l':
                GPIO.output(LEFT, 1)
                GPIO.output(RIGHT, 0)
            elif direction == 'r':
                GPIO.output(LEFT, 0)
                GPIO.output(RIGHT, 1)
            time.sleep(0.1)
            GPIO.output(LEFT, 1)
            GPIO.output(RIGHT, 1)
            time.sleep(0.1)
            GPIO.output(LEFT, 0)
            GPIO.output(RIGHT, 0)
            time.sleep(1)

    except KeyboardInterrupt:
        print("END")
        cap.release()
        GPIO.cleanup()
        
        
        

if __name__ == '__main__':
    main()

