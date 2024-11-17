# import streamlit as st
# import os
# from PIL import Image
# import numpy as np
# import pickle
# import tensorflow as tf
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.layers import GlobalMaxPooling2D
# from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
# from sklearn.neighbors import NearestNeighbors
# from numpy.linalg import norm
# import warnings
#
# warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#
# # Load the precomputed features and filenames
# # filenames.pkl should store actual file paths, not just IDs
# feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
# filenames = pickle.load(open('filenames.pkl', 'rb'))  # Ensure these are valid image paths (strings)
#
# # Initialize the model
# model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# model.trainable = False
# model = tf.keras.Sequential([model, GlobalMaxPooling2D()])
#
# st.title('Fashion Recommender System')
#
# def save_uploaded_file(uploaded_file):
#     """Save the uploaded file to the 'uploads' directory."""
#     try:
#         os.makedirs('uploads', exist_ok=True)
#         file_path = os.path.join('uploads', uploaded_file.name)
#         with open(file_path, 'wb') as f:
#             f.write(uploaded_file.getbuffer())
#         return file_path
#     except:
#         return None
#
# def feature_extraction(img_path, model):
#     """Extract features from an image using the specified model."""
#     img = image.load_img(img_path, target_size=(224, 224))
#     img_array = image.img_to_array(img)
#     expanded_img_array = np.expand_dims(img_array, axis=0)
#     preprocessed_img = preprocess_input(expanded_img_array)
#     result = model.predict(preprocessed_img).flatten()
#     return result / norm(result)
#
# def recommend(features, feature_list):
#     """Get the indices of the 5 most similar images in the dataset."""
#     neighbors = NearestNeighbors(n_neighbors=2, algorithm='brute', metric='euclidean')
#     neighbors.fit(feature_list)
#     _, indices = neighbors.kneighbors([features])
#     return indices
#
# # File upload and recommendation workflow
# uploaded_file = st.file_uploader("Choose an image")
# if uploaded_file is not None:
#     img_path = save_uploaded_file(uploaded_file)
#     if img_path:
#         # Display the uploaded image
#         display_image = Image.open(img_path)
#         st.image(display_image, caption="Uploaded Image", use_container_width=True)
#
#         # Feature extraction for the uploaded image
#         features = feature_extraction(img_path, model)
#
#         # Get recommendations
#         indices = recommend(features, feature_list)
#         print(indices)
#         # Display the recommended images
#         st.write("**Recommended Products:**")
#         cols = st.columns(5)
#         for i, col in enumerate(cols):
#             if i < len(indices[0]):
#                 recommended_img_path = filenames[indices[0][i]]
#
#                 # Check if recommended_img_path is a valid string path
#                 # Assuming filenames contains actual file paths or URLs
#                 if isinstance(recommended_img_path, str) and os.path.exists(recommended_img_path):
#                     recommended_image = Image.open(recommended_img_path)
#                     col.image(recommended_image, use_container_width=True)
#                 else:
#                     st.warning(f"Invalid path: {recommended_img_path}")
#     else:
#         st.error("An error occurred while uploading the file.")

from flask import Flask, request, jsonify
import base64
import pickle
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications import ResNet50
from numpy.linalg import norm
import io
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.Sequential([model, GlobalMaxPooling2D()])

def extract_features(img_data):
    img = Image.open(io.BytesIO(base64.b64decode(img_data)))
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

@app.route('/api/find-similar-images', methods=['POST'])
def find_similar_images():
    try:
        img_data = request.data.decode('utf-8')
        features = extract_features(img_data)

        distances = np.linalg.norm(feature_list - features, axis=1)
        product_distance_pairs = list(zip(filenames, distances))

        sorted_products = sorted(product_distance_pairs, key=lambda x: x[1])
        sorted_keys = [key for key, _ in sorted_products]
        unique_sorted_keys = []
        seen = set()
        for key in sorted_keys:
            if key not in seen:
                unique_sorted_keys.append(key)
                seen.add(key)

        response = [{"product_id": prod} for prod in unique_sorted_keys]
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(port=5000)
