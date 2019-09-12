import tensorflow as tf
from keras import backend as K
from keras.models import load_model
import numpy as np
import os
from sklearn.externals import joblib
from abc import ABC, abstractmethod
import image_utils
import threading
from concurrent.futures import ThreadPoolExecutor
#global context 
global graph

width_image = 224
height_image = 224 
channel = 3 #image channel (red, green, blue)
labels = ["Vetro", "Alluminio", "Carta", "Plastica"] #trash category used to training the net
svm_labels_to_categories = {
    "plastic" : "Plastica",
    "paper" : "Carta",
    "glass" : "Vetro",
    "metal" : "Alluminio"
}
"""
    Image Classifier is used to classify a trash from an image
    an example of prediction can be:
        label = classifier.classify
"""
class ImageClassifier(ABC):
    def __init__(self, model_path):
        super().__init__()
        self.executor = ThreadPoolExecutor(max_workers=1)
        netPromise = self.executor.submit(lambda: self.init(model_path)) #used to load keras net in the same thread where prediction will made
        self.net = netPromise.result() # wait for net loading...
    
    def init(self, model_path):
        return tf.keras.models.load_model(model_path) #load cnn model from file 
    #subit the classify abstract method
    def classify(self, img):
        return self.executor.submit(lambda: self.safeClassify(img)).result() 

    @abstractmethod
    def safeClassify(self, img):
        pass

class MLPClassifier(ImageClassifier):
    #cnn_name is model net stored with h5 extension 
    def __init__(self, cnn_name_h5):
        super().__init__(cnn_name_h5)

    def safeClassify(self, img): 
        image_resized = image_utils.resize_img(img, width_image, height_image)
        tensor = image_utils.img_to_tensor(image_resized)
        max_prob = np.amax(self.net.predict(tensor))
        label_index = self.net.predict_classes(tensor)[0]
        return (labels[label_index], max_prob)

class SVMClassifier(ImageClassifier):
    #cnn_name is model net store with h5 extension and svm_name is the svm name
    def __init__(self, cnn_name_h5, svm_name_sav):
        super().__init__(cnn_name_h5)
        self.svm = joblib.load(svm_name_sav)

    def safeClassify(self, img): 
        image_resized = image_utils.resize_img(img, width_image, height_image)
        tensor = image_utils.img_to_tensor(image_resized)
        feature = self.net.predict(tensor)
        max_prob = np.amax(self.svm.predict_proba(feature))
        svm_label = self.svm.predict(feature)[0]
        return (svm_labels_to_categories[svm_label], max_prob)
