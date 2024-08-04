
# FACIAL-FEATURE BASED IMAGE SORTER

This project allows you to upload photos through a web interface. The photos are processed using a machine learning model, and then sorted into a folder named test.


## Tech Stack

- Flask: Web framework used to create the server.
- Werkzeug: Utility library for securing file uploads.
- OpenCV (cv2): Used for image processing.
- Keras-FaceNet: A pre-trained model for facial recognition.
- MTCNN: Face detection model.
- NumPy: Array processing for numerical data.
- Joblib: For saving and loading models.
- shutil: For high-level file operations.


## Setup and Installation

Follow these steps to set up and run the project

Clone the project

```bash
  https://github.com/Humbledot00/Image-Sorter.git
```

Go to the project directory

```bash
  cd my-project
```
Create the test Folder
```bash
  mkdir test

```

Install dependencies

```bash
  pip install Flask numpy opencv-python keras-facenet mtcnn joblib

```

Start the server

```bash
  python app.py
```

## Project Structure

- app.py: The main application file that starts the Flask server.
- templates/: Contains the HTML templates for the web interface.
- test/: The directory where the processed photos will be sorted.
