import pymysql
import base64
import numpy as np
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications import ResNet50
from numpy.linalg import norm
import pickle
from PIL import Image
import io
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
conn = pymysql.connect(
            host='localhost',
            port=3306,
            user='root',
            password='123456',
            database='3tshop'
        )
cursor = conn.cursor()

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

def extract_features(img_data):
    img = Image.open(io.BytesIO(base64.b64decode(img_data)))
    img = img.convert('RGB')
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

query = "SELECT image_data, product_id FROM Images"
cursor.execute(query)
rows = cursor.fetchall()

feature_list = []
filenames = []

for row in rows:
    img_data = row[0]
    product_id = row[1]

    features = extract_features(img_data)
    feature_list.append(features)
    filenames.append(product_id)

pickle.dump(feature_list, open('embeddings.pkl', 'wb'))
pickle.dump(filenames, open('filenames.pkl', 'wb'))

cursor.close()
conn.close()


