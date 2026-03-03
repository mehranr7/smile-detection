# Smile Detection (HOG, LBP & SVM)

A classical Computer Vision and Machine Learning pipeline designed to detect smiles using the **GENKI-4K** dataset. Instead of relying on Deep Learning (CNNs), this project explicitly extracts manual image features using **HOG (Histogram of Oriented Gradients)** and **LBP (Local Binary Pattern)**, and classifies them using a **Support Vector Machine (SVM)**.

## 📌 Overview

This project demonstrates the core fundamentals of traditional image processing and classification:
1. **Face & Smile Localization:** Uses OpenCV's Haar Cascades to crop faces and isolate the mouth/smile region.
2. **Feature Extraction:** Transforms the raw image pixels into meaningful numerical arrays by calculating shape structures (HOG) and texture properties (LBP).
3. **Classification:** Trains a linear SVM model from `scikit-learn` to differentiate between "Smiling" and "Not Smiling" based on the combined HOG and LBP feature vectors.

## ✨ Key Features

- **Multi-Feature Fusion:** Combines HOG (for edge and shape detection) and LBP (for micro-texture recognition) to create a robust, unified feature vector for every image.
- **Automated Data Preprocessing:** Includes custom Python scripts (`Extract_Parts.py`) to automatically detect, crop, and resize faces and mouths from raw directories, ensuring the SVM receives consistently formatted data.
- **Model Persistence:** Uses `joblib` to save and load the trained SVM model, allowing for fast inference on new test images without needing to retrain the dataset.
- **GENKI-4K Dataset Integration:** Includes a parser to read the official `labels.txt` from the GENKI-4K image database and map them dynamically to the cropped output.

## 🏗️ Architecture & Pipeline

### 1. `Extract_Parts.py`
Utilizes OpenCV (`cv2.CascadeClassifier`) to perform a two-step cropping process: First, it detects the face, and then it detects the smile within that face. It strictly manages aspect ratios to prevent image distortion.

### 2. `Model_Manager.py`
The brain of the project. It handles:
- Reading the original dataset labels.
- Running `skimage.feature.hog` and `local_binary_pattern` on cropped images.
- Splitting the data via `train_test_split`.
- Fitting the `svm.SVC(kernel='linear')` and calculating accuracy metrics.

### 3. `main.py`
The execution entry point. It calculates the elapsed processing time and handles switching between training mode (`Model_Manager.Generate`) and inference mode (`Model_Manager.Load`).

## 🚀 How to Run

1. Clone the repository.
2. Ensure you have the **GENKI-4K** dataset extracted into a folder named `genki4k` in the root directory.
3. Install the required dependencies:
  ```bash
   pip install opencv-python scikit-image scikit-learn numpy joblib
  ```
4. To train the model, uncomment the Model_Manager.Generate() line in main.py.
5. To test the model on new images, place them in the test folder and run:
  ```bash
   python main.py
  ```

## 🛠️ Tech Stack
Language: Python

Machine Learning: scikit-learn (SVM Classifier, Train-Test Split, Accuracy Metrics)

Computer Vision: OpenCV (Haar Cascades) & scikit-image (HOG, LBP)

Data Handling: NumPy, joblib
