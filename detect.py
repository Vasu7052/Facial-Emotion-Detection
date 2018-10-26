from keras.preprocessing import image
from keras.models import model_from_json
import numpy as np

json_file = open('./model/facial_expression_model_structure.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("./model/facial_expression_model_weights.h5")
print("Loaded model from disk")

emotion_dict = {
    0 : 'Angry',
    1 : 'Disgust',
    2 : 'Fear',
    3 : 'Happy',
    4 : 'Sad',
    5 : 'Surprise',
    6 : 'Neutral'
}

img = image.load_img("jackman.png", grayscale=True, target_size=(48, 48))

x = image.img_to_array(img)
print(x.shape)
x = np.expand_dims(x, axis=0)
print(x.shape)
x /= 255
print(x.shape)

custom = model.predict(x)
index_max = np.argmax(custom[0])

predicted = emotion_dict[index_max]
print("Emotion :" , predicted)

