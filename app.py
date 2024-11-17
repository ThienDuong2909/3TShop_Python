# import numpy as np
# import pickle as pkl
# import tensorflow as tf
# from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.layers import GlobalMaxPool2D
#
# from sklearn.neighbors import NearestNeighbors
# import os
# from numpy.linalg import norm
# import streamlit as st
#
# st.header('Fashion Recommendation System')
#
# Image_features = pkl.load(open('Images_features.pkl','rb'))
# filenames = pkl.load(open('filenames.pkl','rb'))
#
# def extract_features_from_images(image_path, model):
#     img = image.load_img(image_path, target_size=(224,224))
#     img_array = image.img_to_array(img)
#     img_expand_dim = np.expand_dims(img_array, axis=0)
#     img_preprocess = preprocess_input(img_expand_dim)
#     result = model.predict(img_preprocess).flatten()
#     norm_result = result/norm(result)
#     return norm_result
# model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
# model.trainable = False
#
# model = tf.keras.models.Sequential([model,
#                                     GlobalMaxPool2D()
#                                     ])
# neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
# neighbors.fit(Image_features)
# upload_file = st.file_uploader("Upload Image")
# if upload_file is not None:
#     with open(os.path.join('upload', upload_file.name), 'wb') as f:
#         f.write(upload_file.getbuffer())
#     st.subheader('Uploaded Image')
#     st.image(upload_file)
#     input_img_features = extract_features_from_images(upload_file, model)
#     distance,indices = neighbors.kneighbors([input_img_features])
#     st.subheader('Recommended Images')
#     col1,col2,col3,col4,col5 = st.columns(5)
#     with col1:
#         st.image(filenames[indices[0][1]])
#     with col2:
#         st.image(filenames[indices[0][2]])
#     with col3:
#         st.image(filenames[indices[0][3]])
#     with col4:
#         st.image(filenames[indices[0][4]])
#     with col5:
#         st.image(filenames[indices[0][5]])
#
# print("hello")
# import numpy as np
# import tensorflow as tf
# import base64
#
# from tensorflow.keras.layers import GlobalMaxPool2D
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
# from tensorflow.keras.models import Model
# from sklearn.neighbors import NearestNeighbors
# import pymysql
# import io
# from PIL import Image
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#
# # Load the pre-trained ResNet50 model (without the top layer for feature extraction)
# base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# x = base_model.output
# x = GlobalMaxPool2D()(x)
# model = Model(inputs=base_model.input, outputs=x)
#
# def preprocess_image(image_data_base64):
#     """Decode base64 image data and preprocess it for ResNet50 model."""
#     # Decode the base64 image
#     img_data = base64.b64decode(image_data_base64)
#     img = Image.open(io.BytesIO(img_data)).convert('RGB')
#     img = img.resize((224, 224))
#
#     # Convert to array and preprocess for ResNet50
#     img_array = np.array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     return preprocess_input(img_array)
#
# def extract_features(image_data_base64):
#     """Extract features from an image using ResNet50 model."""
#     processed_image = preprocess_image(image_data_base64)
#     features = model.predict(processed_image)
#     return features.flatten()
#
# def get_all_image_features():
#     """Retrieve all image data from database and extract features."""
#     # Connect to MySQL
#     connection = pymysql.connect(
#         host='localhost',
#         port='3306',
#         user='root',
#         password='123456',
#         database='3tshop'
#     )
#     cursor = connection.cursor(dictionary=True)
#     cursor.execute("SELECT image_data FROM images")  # Retrieve all image data (base64)
#
#     image_features = []
#     image_ids = []
#
#     for row in cursor.fetchall():
#         image_data = row['image_data']
#         features = extract_features(image_data)
#         image_features.append(features)
#         image_ids.append(row['image_id'])
#
#     connection.close()
#
#     return np.array(image_features), image_ids
#
# def find_similar_images(query_image_base64):
#     """Find the most similar images to the query image."""
#     query_features = extract_features(query_image_base64)
#
#     # Get all image features from the database
#     image_features, image_ids = get_all_image_features()
#
#     # Use NearestNeighbors to find the most similar images
#     nn = NearestNeighbors(n_neighbors=5, metric='cosine')
#     nn.fit(image_features)
#
#     distances, indices = nn.kneighbors([query_features])
#
#     similar_images = []
#     for idx in indices[0]:
#         similar_images.append(image_ids[idx])
#
#     return similar_images
#


# import numpy as np
# import tensorflow as tf
# import base64
# from tensorflow.keras.layers import GlobalMaxPool2D
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
# from tensorflow.keras.models import Model
# from sklearn.neighbors import NearestNeighbors
# import pymysql
# import pymysql.cursors
# import io
# from PIL import Image
# import os
# import json
# from flask import Flask, request, jsonify  # If you're using Flask for the API
#
# app = Flask(__name__)  # Initialize Flask app if using it for API
#
# # Initialize model
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# x = base_model.output
# x = GlobalMaxPool2D()(x)
# model = Model(inputs=base_model.input, outputs=x)
#
#
# def preprocess_image(image_data_base64):
#     img_data = base64.b64decode(image_data_base64)
#     img = Image.open(io.BytesIO(img_data)).convert('RGB')
#     img = img.resize((224, 224))
#     img_array = np.array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     return preprocess_input(img_array)
#
#
# def extract_features(image_data_base64):
#     processed_image = preprocess_image(image_data_base64)
#     features = model.predict(processed_image)
#     return features.flatten()
#
#
# def get_all_image_features():
#     connection = pymysql.connect(
#         host='localhost',
#         port=3306,
#         user='root',
#         password='123456',
#         database='3tshop'
#     )
#     # Use DictCursor to retrieve results as dictionaries
#     cursor = connection.cursor(pymysql.cursors.DictCursor)
#     cursor.execute("SELECT image_id, image_data FROM images")
#     image_features = []
#     image_ids = []
#     for row in cursor.fetchall():
#         image_data = row['image_data']
#         features = extract_features(image_data)
#         image_features.append(features)
#         image_ids.append(row['image_id'])
#     connection.close()
#     return np.array(image_features), image_ids
#
#
# def find_similar_images(query_image_base64):
#     query_features = extract_features(query_image_base64)
#     image_features, image_ids = get_all_image_features()
#
#     # Ensure n_neighbors does not exceed the number of available samples
#     n_neighbors = min(5, len(image_features))
#
#     # Find distances
#     nn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
#     nn.fit(image_features)
#     distances, indices = nn.kneighbors([query_features])
#
#     # Collect and sort results
#     similar_images = [{"image_id": image_ids[idx], "similarity": 1 - distances[0][i]}
#                       for i, idx in enumerate(indices[0])]
#     similar_images.sort(key=lambda x: x["similarity"], reverse=True)  # Sort descending by similarity
#
#     return similar_images
#
#
# @app.route('/find_similar', methods=['POST'])
# def find_similar_endpoint():
#     try:
#         data = request.json
#         query_image_base64 = data['image_data']
#         similar_images = find_similar_images(query_image_base64)
#         return jsonify(similar_images)
#     except Exception as e:
#         print(f"Error: {e}")  # Print error in the console
#         return jsonify({"error": str(e)}), 500
#
#
# if __name__ == '__main__':
#     app.run(port=5000)  # Run on any preferred port
#
# import numpy as np
# import pymysql
# import base64
# from PIL import Image
# import io
# import matplotlib.pyplot as plt
# from pymysql.cursors import DictCursor
#
#
# def get_images_from_db():
#     connection = pymysql.connect(
#         host='localhost',
#         port=3306,
#         user='root',
#         password='123456',
#         database='3tshop'
#     )
#     # Use DictCursor to get dictionary-like rows
#     cursor = connection.cursor(DictCursor)
#     cursor.execute("SELECT image_id, image_data FROM images")
#
#     images = []
#     for row in cursor.fetchall():
#         image_id = row['image_id']
#         image_data = row['image_data']
#
#         # Decode base64 and convert to PIL Image
#         img_data = base64.b64decode(image_data)
#         img = Image.open(io.BytesIO(img_data))
#
#         images.append((image_id, img))
#
#     connection.close()
#     return images
#
#
# def plot_images(images, max_images=5):
#     """Plot a list of images with matplotlib."""
#     plt.figure(figsize=(15, 5))
#     for i, (image_id, img) in enumerate(images[:max_images]):
#         plt.subplot(1, max_images, i + 1)
#         plt.imshow(img)
#         plt.title(f'Image ID: {image_id}')
#         plt.axis('off')
#     plt.show()
#
#
# # Retrieve and plot images
# images = get_images_from_db()
# plot_images(images)

# import tensorflow
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.layers import GlobalMaxPooling2D
# from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
# import numpy as np
# from numpy.linalg import norm
# import os
# from tqdm import tqdm
# import pickle
# import os
#
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# model.trainable = False
#
# model = tensorflow.keras.Sequential([
#     model,
#     GlobalMaxPooling2D()
# ])
#
#
# # print(model.summary())
#
# def extract_features(img_path, model):
#     img = image.load_img(img_path, target_size=(224, 224))
#     img_array = image.img_to_array(img)
#     expanded_img_array = np.expand_dims(img_array, axis=0)
#     preprocessed_img = preprocess_input(expanded_img_array)
#     result = model.predict(preprocessed_img).flatten()
#     normalized_result = result / norm(result)
#
#     return normalized_result
#
#
# filenames = []
#
# for file in os.listdir('images'):
#     filenames.append(os.path.join('images', file))
# # print(os.listdir('images'))
# feature_list = []
#
# for file in tqdm(filenames):
#     feature_list.append(extract_features(file, model))
#
# pickle.dump(feature_list, open('embeddings.pkl', 'wb'))
# pickle.dump(filenames, open('filenames.pkl', 'wb'))

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


