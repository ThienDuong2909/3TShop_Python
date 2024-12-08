from flask import Flask, request, jsonify
import pymysql
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
import os

app = Flask(__name__)

# Load dữ liệu embeddings và filenames
if os.path.exists('embeddings.pkl') and os.path.exists('filenames.pkl'):
    feature_list = pickle.load(open('embeddings.pkl', 'rb'))
    filenames = pickle.load(open('filenames.pkl', 'rb'))
else:
    feature_list = []
    filenames = []

# Load mô hình
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


@app.route('/api/train', methods=['POST'])
def train_model():
    try:
        # Kết nối cơ sở dữ liệu
        conn = pymysql.connect(
            host='localhost',
            port=3306,
            user='root',
            password='123456',
            database='3tshop'
        )
        cursor = conn.cursor()

        # Lấy dữ liệu từ cơ sở dữ liệu
        query = "SELECT image_data, product_id FROM Images"
        cursor.execute(query)
        rows = cursor.fetchall()

        new_features = []
        new_filenames = []

        for row in rows:
            img_data = row[0]
            product_id = row[1]

            # Chỉ thêm ảnh mới
            if product_id not in filenames:
                features = extract_features(img_data)
                new_features.append(features)
                new_filenames.append(product_id)

        # Cập nhật dữ liệu toàn cục
        feature_list.extend(new_features)
        filenames.extend(new_filenames)

        # Lưu lại vào tệp
        pickle.dump(feature_list, open('embeddings.pkl', 'wb'))
        pickle.dump(filenames, open('filenames.pkl', 'wb'))

        cursor.close()
        conn.close()

        return jsonify({
            "message": "Training completed successfully.",
            "new_images_added": len(new_filenames)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/find-similar-images', methods=['POST'])
def find_similar_images():
    try:
        img_data = request.data.decode('utf-8')
        features = extract_features(img_data)

        distances = np.linalg.norm(np.array(feature_list) - features, axis=1)
        distances = distances.astype(float)
        product_distance_pairs = list(zip(filenames, distances))

        sorted_products = sorted(product_distance_pairs, key=lambda x: x[1])
        # sorted_keys = [key for key, _ in sorted_products]
        # unique_sorted_keys = []
        # seen = set()
        # for key in sorted_keys:
        #     if key not in seen:
        #         unique_sorted_keys.append(key)
        #         seen.add(key)
        #
        # response = [{"product_id": prod} for prod in unique_sorted_keys]
        # return jsonify(response)
        product_best_match = {}

        for product_id, distance in sorted_products:
            if product_id not in product_best_match:
                product_best_match[product_id] = distance

        response = [{"product_id": product_id, "distance": product_best_match[product_id]}
                    for product_id in product_best_match]
        print(response)
        return jsonify(response)

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/user-feedback', methods=['POST'])
def user_feedback():
    try:
        data = request.get_json()
        img_data = data['image_data']
        product_id = data['product_id']
        feedback = data['feedback']
        features = extract_features(img_data)
        distances = np.linalg.norm(np.array(feature_list) - features, axis=1)
        distances = distances.astype(float)

        product_distance_pairs = list(zip(filenames, distances))

        matching_products = [pair for pair in product_distance_pairs if pair[0] == product_id]

        if matching_products:
            best_match = min(matching_products, key=lambda x: x[1])
            best_product_id, best_distance = best_match

            if feedback and  0 < best_distance < 0.5 :
                conn = pymysql.connect(
                    host='localhost',
                    port=3306,
                    user='root',
                    password='123456',
                    database='3tshop'
                )
                cursor = conn.cursor()
                query = "INSERT INTO Images (image_data, product_id) VALUES (%s, %s)"
                cursor.execute(query, (img_data, best_product_id))
                conn.commit()

                # Update embeddings
                new_features = extract_features(img_data)
                feature_list.append(new_features)
                filenames.append(best_product_id)

                pickle.dump(feature_list, open('embeddings.pkl', 'wb'))
                pickle.dump(filenames, open('filenames.pkl', 'wb'))

                cursor.close()
                conn.close()

                return jsonify({"message": "Feedback processed and model updated."})
            else:
                return jsonify({"message": "Feedback received but no update to the model."})
        else:
            return jsonify({"message": "No matching product found."})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(port=5000)
