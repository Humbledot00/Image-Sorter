import os
import numpy as np
import cv2
from mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import joblib
import shutil
from tqdm import tqdm

# Mount Google Drive


# Set up paths
# source_folder = 'source_folder'
target_folder = 'test'

# Initialize FaceNet and MTCNN
facenet = FaceNet()
detector = MTCNN()

def load_images_from_folder(folder):
    images = []
    for filename in tqdm(os.listdir(folder), desc="Loading images"):
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
    for filename, img in tqdm(images, desc="Encoding faces"):
        try:
            faces = detect_and_align_faces(img)
            for face in faces:
                encoding = facenet.embeddings(np.expand_dims(face, axis=0))[0]
                encodings.append((filename, encoding))
            if not faces:
                print(f"No faces found in image {filename}")
        except Exception as e:
            print(f"Error encoding image {filename}: {e}")
    return encodings

def cluster_faces(encodings, eps=0.5, min_samples=3):
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
        if filename not in persons[label]:  # Ensure no duplicates
            persons[label].append(filename)
    return persons

def create_folders_and_move_images(persons, source_folder, target_folder):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for person_id, files in persons.items():
        if person_id == -1:  # Skip outliers
            continue

        person_folder = os.path.join(target_folder, f'person_{person_id}')
        if not os.path.exists(person_folder):
            os.makedirs(person_folder)

        for file in files:
            src_file_path = os.path.join(source_folder, file)
            dest_file_path = os.path.join(person_folder, file)
            if not os.path.exists(dest_file_path):
                shutil.copy(src_file_path, dest_file_path)

def train_face_recognition_model(encodings, labels):
    face_encodings = [encoding[1] for encoding in encodings]
    le = LabelEncoder()
    y = le.fit_transform(labels)
    model = SVC(kernel='linear', probability=True)
    model.fit(face_encodings, y)
    return model, le

def startprocessing(source_folder):
    # Main process
    print("Starting face recognition and clustering process...")

    images = load_images_from_folder(source_folder)
    encodings = encode_faces(images)

    if encodings:
        labels = cluster_faces(encodings, eps=0.7, min_samples=3)  # Adjusted eps for FaceNet embeddings

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
            print(f"Total images processed: {len(images)}")
            print(f"Total faces detected and encoded: {len(encodings)}")
            print(f"Number of unique faces detected: {len(set(labels))}")
        else:
            print("Insufficient number of unique faces for training.")
    else:
        print("No faces found in any of the images.")
