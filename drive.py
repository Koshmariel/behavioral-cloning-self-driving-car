from flask import Flask
import socketio
import eventlet
from keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import cv2

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
import tensorflow as tf

sio = socketio.Server()

app = Flask(__name__) #'__main__'

speed_limit = 28
#@app.route('/home')
#def greeting():
#    return 'Welcome'

def img_preprocess(img):
    img = img[60:135,:,:]                      #removing irrelevant parts of the images
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV) #color scheme is more efficien at nvidia model
    img = cv2.GaussianBlur(img, (3,3), 0)      #makes smoother image with less noise
    img = cv2.resize(img, (200, 66))
    img = img/255                              #normalization
    return img


@sio.on('telemetry')
def telemetry(sid, data):
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = img_preprocess(image)
    image = np.array([image]) #changing to 4d
    steering_angle = float(model.predict(image))
#    throttle = 1.0 - speed/speed_limit
    if speed < speed_limit:
        throttle = 1.
    else:
        throttle = 0.
    print('{:7.3f} {:7.3f} {:7.3f}'.format(steering_angle, throttle, speed))
    send_control(steering_angle, throttle)
    

@sio.on('connect')
def connect(sid, eviron):
    print('Connected')
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit('steer', data = {
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    })

if __name__ == '__main__':
#    app.run(port=3000)
    model = load_model('model.h5')
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)