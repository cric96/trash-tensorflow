import os
import sys
from pyzbar.pyzbar import decode
import cv2
import requests
import numpy as np
useCPU = True
if(useCPU):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from flask import Flask, url_for, send_from_directory, request
from flask_cors import CORS, cross_origin
from classifier import MLPClassifier, SVMClassifier
import barcode_classifier 
import image_utils
import result_builder as rb
import tensorflow as tf

CATEGORY_INDEX = 0
PROB_INDEX = 1
PROB_THR = 0.6
#create the classifier
image_classifier = SVMClassifier("model/CNN_SVM_15_40_02_08_2019.h5", "model/CNN_SVM_15_40_02_08_2019.sav")
#image_classifier = MLPClassifier("model/CNN_15_40_02_08_2019.h5")
base_path = "/v0/prediction/"
#create flask app, it is used to start rest api server
app = Flask(__name__)
CORS(app, resources={r"/v0/prediction/*" : {"origins" : "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route(base_path + 'ai', methods = ['POST', 'OPTIONS']) 
@cross_origin()
def api_ai():
    ##app.logger.info(PROJECT_HOME)
    post_data = request.data #read image data
    img = image_utils.load_from_bytes(post_data) #convert image into object type that could be passed to classifier
    prediction = image_classifier.classify(img) #get trash label and probability from image
    print(prediction)
    if(prediction[PROB_INDEX] > PROB_THR): #check if the category predicted has probability greater then the threshould
        return rb.category_found(prediction[CATEGORY_INDEX])
    else:
        return rb.category_not_found()
    
@app.route(base_path + 'barcode', methods = ['POST', 'OPTIONS'])
@cross_origin()
def api_barcode():
    post_data = request.data #read image data
    nparr = np.frombuffer(post_data, np.uint8) #convert image into a format usable in cv2
    ## decode image for cv2 library
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # cv2.IMREAD_COLOR in OpenCV 3.1
    # transform into grayscale image
    gray_img = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    #decode
    barcodes = decode(gray_img)
    if not barcodes:
        return rb.barcode_not_found()
    #the decode is done
    code = barcodes[0].data #take barcode number
    url = "https://it.openfoodfacts.org/api/v0/product/{}.json".format(code) #create the open api url to retrive food information
    data = requests.get(url).json() #send the request and wait the response
    if data["status"] != 1: #if status is not equals to 1 it means that not product is found
        return rb.category_not_found()
    #the product is found
    trash_category = barcode_classifier.get_trash_category(data["product"]) #get trash category from packaging and packaging tag
    if(trash_category != ""):
        return rb.category_found(trash_category)
    else:
        return rb.category_not_found()

if __name__ == '__main__':
    serverPort=7000
    if(len(sys.argv) == 2) :
        serverPort = sys.argv[1]
    app.run(debug=False, port=serverPort, host='0.0.0.0')
