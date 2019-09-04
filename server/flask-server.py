import os
import sys
from pyzbar.pyzbar import decode
import cv2
import numpy as np
useCPU = True
if(useCPU):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from flask import Flask, url_for, send_from_directory, request
from flask_cors import CORS
from classifier import MLPClassifier, SVMClassifier
import image_utils
import http_code
import tensorflow as tf

CATEGORY_INDEX = 0
PROB_INDEX = 1
PROB_THR = 0.9
#create the classifier
image_classifier = SVMClassifier("model/CNN_SVM_15_40_02_08_2019.h5", "model/CNN_SVM_15_40_02_08_2019.sav")
#image_classifier = MLPClassifier("model/CNN_15_40_02_08_2019.h5")

#create flask app, it is used to start rest api server
app = Flask(__name__)
CORS(app)

@app.route('/prediction/ai', methods = ['POST'])

def api_ai():
    ##app.logger.info(PROJECT_HOME)
    post_data = request.data #read image data
    img = image_utils.load_from_bytes(post_data)
    prediction = image_classifier.classify(img)
    if(prediction[PROB_INDEX] > PROB_THR): #check if the category predicted has probability greater then the threshould
        return {'trashCategory': prediction[CATEGORY_INDEX]}, http_code.ok
    else:
        return {}, http_code.no_category
    
@app.route('/prediction/barcode', methods = ['POST'])
def api_barcode():
    #TODO
    # retrieve image from request
    # call prediction on barcode class (or function)
    # create a request from the open food api
    # retrieve the category (if it is found), code 200
    # if it isn't found a category, return empty json and code 201
    # if it isn't found a barcode, return empty json and code 202
    post_data = request.data #read image data
    nparr = np.frombuffer(post_data, np.uint8)
    ## decode image for cv2 library
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # cv2.IMREAD_COLOR in OpenCV 3.1
    gray_img = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    #decode
    barcodes = decode(gray_img)
    if barcodes:
        barcode_found = barcodes[0].data.decode('utf-8')
    return ''

if __name__ == '__main__':
    serverPort=8080
    if(len(sys.argv) == 2) :
        serverPort = sys.argv[1]
    app.run(debug=False, port=serverPort, host='0.0.0.0')
