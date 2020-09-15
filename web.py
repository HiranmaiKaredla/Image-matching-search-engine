import warnings  
with warnings.catch_warnings():
  warnings.filterwarnings("ignore",category=FutureWarning)


import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor
from datetime import datetime
from flask import Flask, request, render_template
#from pathlib import Path
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
import os
import glob

app = Flask(__name__)
app.config['DEBUG'] = True
#with tf.Session() as sess:
#tf.reset_default_graph()
#global graph
#graph = tf.Graph()
dir_path = os.path.dirname(os.path.realpath(__file__))

global graph
sess = tf.compat.v1.Session()
graph = tf.compat.v1.get_default_graph()
init = tf.global_variables_initializer()
sess.run(init)
set_session(sess)
model = load_model(dir_path+'/model/vgg19.h5')
def predict(img):
  img = img.resize((224, 224))
  img = img.convert('RGB')
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
  feature = model.predict(x)[0]
  return feature / np.linalg.norm(feature)


features = []
img_paths = []
for each in glob.glob(dir_path+'/features/*.npy'):
  f = np.load(each)
  #print(count, each, 1-distance.cosine(f,feature))
  img_paths.append("/static/data/"+(each.rsplit('/',1)[-1]).rsplit('.',1)[0] + ".jpg")
  features.append(f)
  

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img']

        # Save query image
        img = Image.open(file.stream)  # PIL image
        uploaded_img_path = "static/upload/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
        img.save(uploaded_img_path)

        # Run search
        #fe = FeatureExtractor()
        #with g.as_default():
        #query = fe.extract(img)
        with graph.as_default():
          set_session(sess)
          query = predict(img)
        dists = np.linalg.norm(features-query, axis=1)  # L2 distances to features
        ids = np.argsort(dists)[:30]  # Top 30 results
        scores = [(dists[id], img_paths[id]) for id in ids if dists[id] < 1.1]

        return render_template('index.html',
                               query_path=uploaded_img_path,
                               scores=scores)
    else:
        return render_template('index.html')


if __name__== "__main__":
  app.run(port =5000)