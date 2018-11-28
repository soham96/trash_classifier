import keras
from pypylon import pylon
from keras.models import Model, load_model
from keras.applications import mobilenet
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing import image
from keras.utils.generic_utils import CustomObjectScope
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2


def pp_image(img):
    img = image.load_img('pic.png', target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    return np.asarray(x)


model=load_model('models/model1.h5', custom_objects={'relu6': mobilenet.relu6})

camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

numberOfImagesToGrab = 100
camera.StartGrabbingMax(numberOfImagesToGrab)

converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

while camera.IsGrabbing():
    time.sleep(0.005)
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if grabResult.GrabSucceeded():
        # Access the image data.
        print("SizeX: ", grabResult.Width)
        print("SizeY: ", grabResult.Height)
        import ipdb; ipdb.set_trace()
        img = converter.Convert(grabResult).GetArray()
        cv2.imwrite('pic.png', img)
        img=pp_image(img)
        yo=model.predict(img)
        print("Gray value of first pixel: ", img[0, 0])
    
    grabResult.Release()
