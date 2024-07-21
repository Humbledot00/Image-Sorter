import os
import numpy as np
import cv2
from keras_facenet import FaceNet
from mtcnn import MTCNN
import joblib
import shutil 

# Initialize FaceNet and MTCNN
facenet = FaceNet()
detector = MTCNN()

# Set up paths
model_path = 'sorted_images/face_recognition_model.pkl'
le_path = 'sorted_images/label_encoder.pkl'

# Load the trained model and label encoder
model = joblib.load(model_path)
label_encoder = joblib.load(le_path)

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        try:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append((filename, img))
        except Exception as e:
            print(f"Error loading image {filename}: {e}")
    return images

def detect_and_align_faces(img):
    faces = detector.detect_faces(img)
    aligned_faces = []
    for face in faces:
        x, y, width, height = face['box']
        face_img = img[y:y+height, x:x+width]
        face_img = cv2.resize(face_img, (160, 160))
        aligned_faces.append(face_img)
    return aligned_faces

def encode_faces(images):
    encodings = []
    for filename, img in images:
        faces = detect_and_align_faces(img)
        for face in faces:
            encoding = facenet.embeddings(np.expand_dims(face, axis=0))[0]
            encodings.append((filename, encoding))
    return encodings

def predict_faces(encodings):
    predictions = {}
    for filename, encoding in encodings:
        if encoding is not None:
            prediction = model.predict([encoding])
            predicted_label = label_encoder.inverse_transform(prediction)[0]
            predictions[filename] = predicted_label
        else:
            predictions[filename] = "Unknown"  # For cases with no valid encoding
    return predictions

def process_images(upload_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load images from the upload folder
    images = load_images_from_folder(upload_folder)
    encodings = encode_faces(images)

    # Predict faces
    predictions = predict_faces(encodings)

    # Save sorted images to the output folder
    for filename, label in predictions.items():
        source_path = os.path.join(upload_folder, filename)
        destination_folder = os.path.join(output_folder, label)
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
        shutil.move(source_path, os.path.join(destination_folder, filename))

    return predictions
