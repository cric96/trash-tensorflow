import tensorflow as tf 
import numpy
import os
from sklearn.externals import joblib
from abc import ABC, abstractmethod
import utils

width_image = 224
height_image = 224
labels = ["cartone", "vetro", "metallo", "carta", "plastica"]
class ImageClassifier(ABC):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def classify(self, img):
        pass

class MLPClassifier(ImageClassifier):
    #cnn_name is model net stored with h5 extension 
    def __init__(self, cnn_name_h5):
        super().__init__()
        self.net = tf.keras.models.load_model(cnn_name_h5)

    def classify(self, img): 
        resized = utils.resize_img(img, width_image, height_image)
        tensor = utils.img_to_tensor(resized)
        labex_index = self.net.predict_classes(tensor)[0]
        return labels[labex_index]

class SVMClassifier(ImageClassifier):
    #cnn_name is model net store with h5 extension and svm_name is the svm name
    def __init__(self, cnn_name_h5, svm_name_sav):
        super().__init__()
        self.net = tf.keras.models.load_model(cnn_name_h5)
        self.svm = joblib.load(svm_name_sav)
        self.net = tf.keras.models.Model(self.net.input, self.net.layers[-2].output)

    def classify(self, img): 
        resized = utils.resize_img(img, width_image, height_image)
        tensor = utils.img_to_tensor(resized)
        feature = self.net.predict(tensor)
        return self.svm.predict(feature)[0]