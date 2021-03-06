import io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from keras.preprocessing import image
from skimage.transform import resize
import numpy as np

def load_from_bytes(img_bytes):
    img = io.BytesIO(img_bytes) 
    img = image.load_img(img)
    return img

def resize_img(img, width, height):
    return img.resize((height, width), Image.BOX)

def img_to_tensor(img):
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]
    return img_tensor            