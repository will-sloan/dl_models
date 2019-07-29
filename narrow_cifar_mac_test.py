from tensorflow.keras.models import load_model
import numpy as np
from keras.preprocessing.image import load_img, img_to_array

model = load_model("narrow_model.h5")
image_path = "C:/Users/OWNER/Desktop/Mac Photos/mac photo 4.jpg"
img = load_img(image_path, target_size=(32,32))
img_array = img_to_array(img)
print(img_array.shape)
img_array = np.expand_dims(img_array, axis=0)
print(img_array.shape)
prediction = model.predict(img_array)
prediction
