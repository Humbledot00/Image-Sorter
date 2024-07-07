# -*- coding: utf-8 -*-
"""Untitled11.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1B49T-UXlP3r3AbC5n_51ssCUw5LrjJD6
"""

!pip install face_recognition
!apt-get install -y libsm6 libxext6 libxrender-dev
!pip install opencv-python

import os
import numpy as np
import cv2
import face_recognition
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import shutil
from google.colab import drive
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

import os
import numpy as np
import cv2
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import joblib
import shutil
from google.colab import drive
from tqdm import tqdm

# Mount Google Drive
drive.mount('/content/drive')

# Set up paths
source_folder = '/content/drive/MyDrive/Images'
target_folder = '/content/drive/MyDrive/sorted_images'

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def load_image(img_path):
    return cv2.imread(img_path)

def load_images_from_folder(folder):
    images = []
    for filename in tqdm(os.listdir(folder), desc="Loading images"):
        img_path = os.path.join(folder, filename)
        try:
            img = load_image(img_path)
            if img is not None:
                images.append((filename, img))
        except Exception as e:
            print(f"Error loading image {filename}: {e}")
    return images

def encode_faces(images):
    encodings = []
    for filename, img in tqdm(images, desc="Encoding faces"):
        try:
            face_locations = face_recognition.face_locations(img, model="hog")
            if face_locations:
                face_encodings = face_recognition.face_encodings(img, face_locations)
                for i, face_encoding in enumerate(face_encodings):
                    encodings.append((f"{filename}_face_{i}", face_encoding))
            else:
                print(f"No faces found in image {filename}")
        except Exception as e:
            print(f"Error encoding image {filename}: {e}")

    print(f"Total faces detected and encoded: {len(encodings)}")
    return encodings

# After clustering
print(f"Number of unique faces detected: {len(set(labels))}")
# The rest of the functions (cluster_faces, sort_images_by_person, create_folders_and_move_images, train_face_recognition_model)
# remain the same as in Version 1

# Main process
print("Starting face recognition and clustering process...")

images = load_images_from_folder(source_folder)
encodings = encode_faces(images)

# The rest of the main process remains the same as in Version 1

def cluster_faces(encodings, eps=0.5, min_samples=2):
    face_encodings = [encoding[1] for encoding in encodings]
    if len(face_encodings) == 0:
        return np.array([])
    labels = DBSCAN(metric='euclidean', eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(face_encodings)
    return labels

def sort_images_by_person(encodings, labels):
    persons = {}
    for idx, label in enumerate(labels):
        filename, encoding = encodings[idx]
        if label not in persons:
            persons[label] = []
        persons[label].append(filename)
    return persons


def create_folders_and_move_images(persons, source_folder, target_folder):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for person_id, files in persons.items():
        person_folder = os.path.join(target_folder, f'person_{person_id}')
        if not os.path.exists(person_folder):
            os.makedirs(person_folder)

        for file in files:
            src_file_path = os.path.join(source_folder, file.split('_face_')[0])
            dest_file_path = os.path.join(person_folder, file.split('_face_')[0])
            if not os.path.exists(dest_file_path):
                shutil.copy(src_file_path, dest_file_path)


def train_face_recognition_model(encodings, labels):
    face_encodings = [encoding[1] for encoding in encodings]
    le = LabelEncoder()
    y = le.fit_transform(labels)
    model = SVC(kernel='linear', probability=True)
    model.fit(face_encodings, y)
    return model, le


if encodings:
    labels = cluster_faces(encodings)

    if len(set(labels)) > 1:
        persons = sort_images_by_person(encodings, labels)
        create_folders_and_move_images(persons, source_folder, target_folder)

        model, label_encoder = train_face_recognition_model(encodings, labels)

        model_path = os.path.join(target_folder, 'face_recognition_model.pkl')
        le_path = os.path.join(target_folder, 'label_encoder.pkl')
        joblib.dump(model, model_path)
        joblib.dump(label_encoder, le_path)


        print(f"Process completed. Sorted images are in {target_folder}")
        print(f"Model saved at {model_path}")
        print(f"Label encoder saved at {le_path}")
    else:
        print("Insufficient number of unique faces for training.")
else:
    print("No faces found in any of the images.")