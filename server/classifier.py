import tensorflow as tf
from keras import backend as K
from keras.models import load_model
import numpy as np
import os
from sklearn.externals import joblib
from abc import ABC, abstractmethod
import image_utils
#global context 
global graph

width_image = 224
height_image = 224
channel = 3
labels = ["cartone", "vetro", "metallo", "carta", "plastica"]
class ImageClassifier(ABC):
    def __init__(self, model_path):
        super().__init__()
        self.net = tf.keras.models.load_model(model_path)
        #self.net.predict(np.array([[0],[0],[0],[0]])) # warmup
        self.net._make_predict_function()
        self.net._make_test_function()
        self.net._make_train_function()
        K.manual_variable_initialization(True)
        #K.clear_session()
        #graph.finalize()
        self.net.predict(np.zeros([4, width_image,height_image,channel],dtype=np.uint8))

    def query(self, action_on_safe_thread):
        #run on global
        self.net._make_predict_function()
        prediction = action_on_safe_thread(self.net)
        return prediction

    @abstractmethod
    def classify(self, img):
        pass

class MLPClassifier(ImageClassifier):
    #cnn_name is model net stored with h5 extension 
    def __init__(self, cnn_name_h5):
        super().__init__(cnn_name_h5)

    def classify(self, img): 
        resized = image_utils.resize_img(img, width_image, height_image)
        tensor = image_utils.img_to_tensor(resized)
        #label_index = self.net.predict_classes(tensor)[0]
        label_index = self.query(lambda net: net.predict_classes(tensor)[0])
        return labels[label_index]

class SVMClassifier(ImageClassifier):
    #cnn_name is model net store with h5 extension and svm_name is the svm name
    def __init__(self, cnn_name_h5, svm_name_sav):
        super().__init__(cnn_name_h5)
        self.svm = joblib.load(svm_name_sav)
        self.net = tf.keras.models.Model(self.net.input, self.net.layers[-2].output)

    def classify(self, img): 
        resized = image_utils.resize_img(img, width_image, height_image)
        tensor = image_utils.img_to_tensor(resized)
        #feature = self.query(tensor)
        feature = self.query(lambda net: net.predict(tensor))
        return self.svm.predict(feature)[0]