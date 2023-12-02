#!/usr/bin/env python
# coding: utf-8

# ## DOWNLOADING THE MODEL

import numpy as np
import os
import tflite_runtime.interpreter as tflite
from io import BytesIO
from urllib import request
from PIL import Image

MODEL_NAME = os.getenv('MODEL_NAME', 'dino-vs-dragon-v2.tflite')

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def prepare_input(image):
    x=np.array(image,dtype='float32')
    return (np.array([x])/255.0)


interpreter = tflite.Interpreter(model_path=MODEL_NAME)
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

def predict(url):
    img = download_image(url)
    img = prepare_image(img, target_size=(150, 150))
    X = prepare_input(img)

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()

    preds = interpreter.get_tensor(output_index)
    return float(preds[0, 0])

def lambda_handler(event, context):
    url = event['url']
    pred = predict(url)
    result = {
        'prediction': pred
    }
    return result


