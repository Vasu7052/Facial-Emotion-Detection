from keras.preprocessing import image
from keras.models import model_from_json
import numpy as np
from keras.models import load_model

gender_classifier = load_model('./model/gender_model.hdf5', compile=False)

gender_dict = {
    0 : 'Female',
    1 : 'Male'
}

img = image.load_img("jackman.png", color_mode='rgb', target_size=(48, 48))

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x /= 255

print(x.shape)

gender_prediction = gender_classifier.predict(x)
gender_label_arg = np.argmax(gender_prediction)
gender_text = gender_dict[gender_label_arg]

print("Gender :" , gender_text)

