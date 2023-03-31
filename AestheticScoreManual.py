import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from keras.applications.inception_v3 import InceptionV3
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from PIL import Image
import PIL
from glob import glob
import re
import requests
import os
import logging
import json

weight_path_trained = './model/Aesthetics_22_March.h5'

weight_path = r'./model/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
IMG_SHAPE = (299, 299, 3)
base_model = InceptionV3(weights = weight_path ,
                         input_shape=IMG_SHAPE,
                         include_top=False)
base_model.trainable = False

data_augmentation = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
  tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])

preprocess_input = tf.keras.applications.inception_v3.preprocess_input
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
  tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(1)

inputs = tf.keras.Input(shape=(299, 299, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

model.compile(optimizer= tf.keras.optimizers.Adam(lr= 0.0001),
              loss= tf.keras.losses.BinaryCrossentropy(from_logits= True),
              metrics= ['accuracy'])

model.load_weights(weight_path_trained)
model.summary()

print("___InceptionV3 Imported!___")

def img_to_vector(img):
    img = img.resize((299,299))
    np_img = np.array(img)
    xx = []
    xx.append(np_img)
    xx = np.array(xx)
    return xx

def enter_log(text):
  f = open(r"Aesthetic_logfile.txt", "a")  
  f.write(text+"\n") 
  f.close()
  
def send_slack(text):
  s = "https://slack.com/api/chat.postMessage?token=xoxp-191928386964-191976206613-362151076802-76c01eca4e9dd2062cbb0b0eaa67d409&channel=%23scraping&text="
  requests.post(s + text)
  
'''
Aesthetics for only one image
'''
def get_asthetic_score(img_url):
  try:
    img_req = requests.get(img_url, stream=True)
    if img_req.status_code != 200:
      return False
    img__= Image.open(img_req.raw)
    imageV = img_to_vector(img__)
    predictions = model.predict(imageV).flatten()
    predictions = tf.nn.sigmoid(predictions)
    return round((1 - predictions.numpy()[0])*100,2)
  except Exception as e:
    print("AESTHETIC EXCEPTION URL: {}".format(str(e)))
    enter_log("AESTHETIC EXCEPTION URL: {}".format(str(e)))
    send_slack("AESTHETIC EXCEPTION URL: {}".format(str(e)))
    return False


'''
Main Code
'''
# get_url = 'http://44.229.68.155/insta_user/get_insta_posts_links?last_post_id=38485363&limit=50'
get_url = 'http://44.229.68.155/insta_user/get_insta_posts_links?offset=0'
post_url = 'http://44.229.68.155/insta_user/update_aesthetic_score'
update_url = 'http://44.229.68.155/insta_user/update_aesthetic_score?update_to_insta_user=1'

x = requests.get(get_url, headers={'Authorization': 'Token ruor7REQi9KJz6wIQKDXvwtt'})
status = x.status_code
data = x.json()
post_count = 0
page = 0
offset = 0
while data['status'] == True:
    try:
        offset = int(data['offset'])
        page_post_count = 0
        aesthetic_details = []
        if len(data['post_links']) == 0: # No data condition
          x = requests.get('http://44.229.68.155/insta_user/get_insta_posts_links?offset={}'.format(offset + 1), headers={'Authorization': 'Token ruor7REQi9KJz6wIQKDXvwtt'})
          status = x.status_code
          data = x.json()
          print("Skipped page due to no post link {}".format(page))
          enter_log("Skipped page due to no post link {}".format(page))
          continue
        last_post_id = data['post_links'][-1]['insta_user_post_id']
        last_insta_user_id = data['post_links'][-1]['insta_user_id']
        for posts in data['post_links']:
            insta_user_id = posts['insta_user_id']
            insta_user_post_id = posts['insta_user_post_id']
            if len(posts['post_link']) < 70:  # Broken link condition
              continue
            image_aesthetic_score = get_asthetic_score(posts['post_link'])
            if image_aesthetic_score == False:
              continue
            aesthetic_details.append({'insta_user_id':insta_user_id,
                                      'insta_user_post_id':insta_user_post_id,
                                      'aesthetic_score':image_aesthetic_score})
            post_count = post_count + 1
            page_post_count = page_post_count + 1
            print("Appended {} posts in {} page with insta_user {} and post_id {}".format(page_post_count, page, insta_user_id, insta_user_post_id))
            enter_log("Appended {} posts in {} page with insta_user {} and post_id {}".format(page_post_count, page, insta_user_id, insta_user_post_id))
        page = page + 1
        offset = offset + 1
        if len(aesthetic_details) == 0:
          x = requests.get('http://44.229.68.155/insta_user/get_insta_posts_links?offset={}'.format(offset), headers={'Authorization': 'Token ruor7REQi9KJz6wIQKDXvwtt'})
          status = x.status_code
          data = x.json()
          print("Skipped page {}".format(page))
          enter_log("Skipped page {}".format(page))
          continue
        # POST the results
        response = {
          "status": True,
          "results": json.dumps(aesthetic_details)
        }
        y = requests.post(post_url, data = response, headers={'Authorization': 'Token ruor7REQi9KJz6wIQKDXvwtt'})
        if y.status_code == 200:
          print("Posted page {} with last insta_user {}".format(page,insta_user_post_id))
          enter_log("Posted page {} with last insta_user {}".format(page,insta_user_post_id))
        else:
          print("Error {} in posting page {}".format(y.status_code, page -1))
          enter_log("Error {} in posting page {}".format(y.status_code, page -1))
          send_slack("AESTHETICS POST Error {} in posting page {}".format(y.status_code, page -1))
        x = requests.get('http://44.229.68.155/insta_user/get_insta_posts_links?offset={}'.format(offset), headers={'Authorization': 'Token ruor7REQi9KJz6wIQKDXvwtt'})
        status = x.status_code
        data = x.json()
        print("Page {} is done with last_post_id {} and InstaUser {}".format(page, last_post_id, insta_user_post_id))
        enter_log("Page {} is done with last_post_id {} and InstaUser {}".format(page, last_post_id, insta_user_post_id))
    except Exception as e:
        print("AESTHETICS EXCEPTION: {}".format(str(e)))
        enter_log("AESTHETICS EXCEPTION: {}".format(str(e)))
        send_slack("AESTHETICS EXCEPTION: {}".format(str(e)))
        x = requests.get('http://44.229.68.155/insta_user/get_insta_posts_links?offset={}'.format(offset+1), headers={'Authorization': 'Token ruor7REQi9KJz6wIQKDXvwtt'})
        status = x.status_code
        data = x.json()
    
if data['status'] == False and data['status'] != True:
    print("-----------------\n All posts are updated (Aesthetic Scores) \n-----------------\n last_post_id: {}\n Total Posts: {}".format(last_post_id, post_count))
    enter_log("-----------------\n All posts are updated (Aesthetic Scores) \n-----------------\n last_post_id: {}\n Total Posts: {}".format(last_post_id, post_count))
else:
    print("Error occured: ", x.status_code)
    enter_log("Error occured: ", x.status_code)
if post_count > 0:
  print("Updating InstaUser...")
  y = requests.post(update_url, headers={'Authorization': 'Token ruor7REQi9KJz6wIQKDXvwtt'})
  if y.status_code == 200:
    if y.json()['status'] == True:
      print("InstaUser is updated!")
      enter_log("InstaUser is updated!")
  else:
    print("Error occured in updating InstaUser")
    enter_log("Error occured in updating InstaUser")
