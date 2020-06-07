#Import libraries
import argparse
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import numpy as np
import json
from PIL import Image

import warnings
warnings.filterwarnings('ignore')

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)


#Add arguments
parser = argparse.ArgumentParser(description="TensorFlow Image Classifier")
parser.add_argument('image', help="Input image file", type=str)
parser.add_argument('model', help="Input Keras model saved", type=str)
parser.add_argument('--top_k', help="Input the top number of restults that you desire", default=1, type=int)
parser.add_argument('--category_names', help="Input the JSON file that contains mapping labels", type=str)


args = parser.parse_args()

#Main functions

load_model = tf.keras.experimental.load_from_saved_model(args.model, custom_objects={'KerasLayer':hub.KerasLayer})
my_image = args.image
k_num = args.top_k
json_file = args.category_names


def process_image(image, size=224):
    image= tf.cast(image, tf.float32) #Change dtype
    image= tf.image.resize(image, [size,size])
    image /= 255 #noramlize pixels
    return image

def predict(image_path, model, top_k):
    im = Image.open(image_path)
    to_array = np.asarray(im)
    processed_image = process_image(to_array)
    ps = model.predict(np.expand_dims(processed_image, axis=0))
    
    probs = -np.sort(-ps[0])[:top_k]
    classes = np.argsort(-ps[0])[:top_k]

    return probs, classes, processed_image

#Return results
probs, classes, processed_image = predict(my_image,load_model, k_num)

#Return these if script is running by itself
if __name__ == '__main__':  
    #Output depending whether json file is provided
    if json_file != None:
        with open(json_file, 'r') as f:
            class_names = json.load(f)

        name_list = []
        classes += 1

        for i in classes:
            name_list.append(class_names[str(i)])

        print('The top most likely class values:', name_list)
        print('Corresponding probabilities:', probs)

    else:
        print('The top most likely classes:', classes)
        print('Corresponding probabilities:', probs)

