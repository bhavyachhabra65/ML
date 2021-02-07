import cv2
import sys
import json
import time
import numpy as np
import keras
from keras.models import model_from_json
import h5py

emotion_labels = ['angry', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# load json and create model arch


f = h5py.File('Model.h5', 'r')
print(f)
print(f.attrs.get('keras_version'))
json_file = open('model.json','r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# load weights into new model
model.load_weights('model.h5')

def predict_emotion(face_image_gray): # a single cropped face
    resized_img = cv2.resize(face_image_gray, (48,48), interpolation = cv2.INTER_AREA)
    # cv2.imwrite(str(index)+'.png', resized_img)
    image = resized_img.reshape(1, 1, 48, 48)
    list_of_list = model.predict(image, batch_size=1, verbose=1)
    angry, fear, happy, sad, surprise, neutral = [prob for lst in list_of_list for prob in lst]
    return [angry, fear, happy, sad, surprise, neutral]


# img_gray = cv2.imread('54.jpg')
# img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
# angry, fear, happy, sad, surprise, neutral = predict_emotion(img_gray)
# for i in [angry,fear,happy,sad,surprise,neutral]:
#     print(i)
# print(([angry,fear,happy,sad,surprise,neutral]).index(max([angry,fear,happy,sad,surprise,neutral])))
# print([angry,fear,happy,sad,surprise,neutral][([angry,fear,happy,sad,surprise,neutral]).index(max([angry,fear,happy,sad,surprise,neutral]))])


cascPath = 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)


jpg_file = sys.argv[1]
img_gray = cv2.imread(jpg_file)
img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)


faces = faceCascade.detectMultiScale(
        img_gray,
        scaleFactor=1.1,
        minNeighbors=1,# 
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
def return_emotion(lis):
    max_prob=max(lis)
    dic={0:'angry',1:'fear',2:'happy',3:'sad',4:'surprise',5:'neutral'}
    return dic[lis.index(max_prob)]


for (x, y, w, h) in faces:
    face_image_gray = img_gray[y:y+h, x:x+w]
    # cv2.rectangle((x, y), (x+w, y+h), (0, 255, 0), 2)
    
    angry, fear, happy, sad, surprise, neutral = predict_emotion(face_image_gray)
    print(return_emotion([angry, fear, happy, sad, surprise, neutral]))
