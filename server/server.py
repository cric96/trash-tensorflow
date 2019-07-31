from http.server import BaseHTTPRequestHandler, HTTPServer

import os
useCPU = True
if(useCPU):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import json
import base64
import io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from classifier import MLPClassifier, SVMClassifier
import utils
##CAN DO BETTER?
#image_classifier = MLPClassifier("adam_densnet121_fine_tuning.h5")
image_classifier = SVMClassifier("adam_densnet121_fine_tuning.h5", "ccn_svm.sav") 
#image_classifier = Fake()

class ClassifierHttpServer(BaseHTTPRequestHandler):
    def _set_response(self): #base header resposnse
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

    def do_POST(self): #post handler
        print("reiceved...")
        content_length = int(self.headers['Content-Length']) 
        post_data = self.rfile.read(content_length) #read image data
        img = utils.load_from_bytes(post_data)
        label = image_classifier.classify(img)
        self._set_response()
        self.wfile.write(json.dumps({"trashClass": label}).encode("utf-8"))
        
    def do_OPTIONS(self): #used due browser problem, first browser send OPTIONS to server!
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Credentials", "true")
        self.send_header("Access-Control-Allow-Methods", "GET,HEAD,OPTIONS,POST,PUT")
        self.send_header("Access-Control-Allow-Headers",
                           "Access-Control-Allow-Headers, Origin,Accept, X-Requested-With, Content-Type, Access-Control-Request-Method, Access-Control-Request-Headers")
        print("options reiceved..")
        self.end_headers()

def run():
    ##make parametrizable 
    print('starting server...')
    server_address = ('192.168.178.101', 8080) 
    httpd = HTTPServer(server_address, ClassifierHttpServer)
    print('running server...')
    httpd.serve_forever()
run()