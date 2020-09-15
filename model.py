from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.models import Model
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
base_model = VGG19(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

model.save(dir_path+'/model/vgg19.h5')