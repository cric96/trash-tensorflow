import os
useCPU = True
if(useCPU):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from flask import Flask, url_for, send_from_directory, request
from flask_cors import CORS
from classifier import MLPClassifier, SVMClassifier
import tensorflow as tf
import utils
##CAN DO BETTER?
#image_classifier = MLPClassifier("adam_densnet121_fine_tuning.h5")
image_classifier = SVMClassifier("adam_densnet121_fine_tuning.h5", "ccn_svm.sav") 

app = Flask(__name__)
CORS(app)

@app.route('/', methods = ['POST'])
def api_root():
    print("reiceved")
    ##app.logger.info(PROJECT_HOME)
    post_data = request.data #read image data
    img = utils.load_from_bytes(post_data)
    label = image_classifier.classify(img)
    return {'trashClass': label}, 200

if __name__ == '__main__':
    app.run(debug=True, port=8080)