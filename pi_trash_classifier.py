import keras
from picamera import PiCamera
from picamera.array import PiRGBArray

from keras.models import Model, load_model
from keras.applications import mobilenet
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing import image
from keras.utils.generic_utils import CustomObjectScope
import numpy as np
#import matplotlib.pyplot as plt
import time
import cv2
import os

def pp_image():
    img = image.load_img('pic.png', target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    return np.asarray(x)

prediction_list=['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
model=load_model('models/model1.h5', custom_objects={'relu6': mobilenet.relu6})

camera = PiCamera()
rawCapture=PiRGBArray(camera)

for i in range(10):
    time.sleep(0.5)

    try:
        import ipdb; ipdb.set_trace()
        # Access the image data.
        camera.capture(rawCapture, format='rgb')
        img=rawCapture.array
        cv2.imwrite('pic.png', img)
        #import ipdb; ipdb.set_trace()
        pred_img=pp_image()
        yo=model.predict(pred_img)
        pred=prediction_list[np.argmax(yo)]
        cv2.putText(img, pred, (10,1000), cv2.FONT_HERSHEY_SIMPLEX, 5, (0,0,0), 5, False)
        name='img'+str(i)+'.png'
        cv2.imwrite(os.path.join('prediction_images', name), img)
        rawCapture.truncate(0)
        #print("Gray value of first pixel: ", img[0, 0])
    except:
        print('Could not perform prediction')

camera.stop_preview()
