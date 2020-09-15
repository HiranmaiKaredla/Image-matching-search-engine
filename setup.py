import os
from feature_extractor import FeatureExtractor
import feature_extractor as fe
from PIL import Image
import glob
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))
data_dir = dir_path+'/'+"static/data"
feature_dir = dir_path+'/'+"features"
os.makedirs(data_dir, exist_ok = True)
os.makedirs(feature_dir, exist_ok = True)

f = FeatureExtractor()
#f.store_features(data_dir , feature_dir)
#fe.store_features(data_dir , feature_dir)

data_path = data_dir
storing_path = feature_dir
for each in glob.glob(data_path+'/*'):
	image_name = (each.rsplit('/',1)[-1]).rsplit('.jpg')[0]
	img = Image.open(each)
	feature = f.extract(img)
	np.save(storing_path+'/'+image_name, feature)

