from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import glob

class FeatureExtractor:
    def __init__(self):
        base_model = VGG19(weights='imagenet')
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

    def extract(self, img):
    	#tf.reset_default_graph()
    	img = img.resize((224, 224))
    	img = img.convert('RGB')
    	x = image.img_to_array(img)
    	x = np.expand_dims(x, axis=0)
    	x = preprocess_input(x)
    	feature = self.model.predict(x)[0]
    	return feature / np.linalg.norm(feature)