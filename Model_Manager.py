from skimage.feature import hog, local_binary_pattern
import numpy as np
import cv2
import os
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import Extract_Parts

# read genki4 lables file
def Read_labels():
    f = open("labels.txt", "r")
    features = np.empty(0)
    label = np.empty(0)
    row = f.readline()
    while row != "":
        col = row.split()
        label = np.append(label,col[0])
        feature = np.empty(0)
        for c in range(1,len(col)):
            feature = np.append(feature,col[c])
        features = np.append(features,feature)
        row = f.readline()
    return label, features

# get hog features of an image
def Get_HOG_singular(image,orientations, pixcels_per_cell):
    fd, hog_image = hog(image, orientations=orientations, pixels_per_cell=(pixcels_per_cell, pixcels_per_cell), cells_per_block=(1, 1), visualize=True, channel_axis=-1)
    return fd

# get lbp features of an image
def Get_LBP_singular(image, P, R):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P, R)
    return lbp.flatten()

# get index of a sample
def Get_sample_index(filename):
    file_part = filename.split("-")[0]
    number_part = file_part[4:]
    return int(number_part) - 1

# get x_train and y_train
def Get_labels_features(source_folder_path, faces_folder_path, smiles_folder_path, get_HOG, get_LBP, is_test = False, orientations = 8, pixcels_per_cell = 4, P = 1, R = 8):
    
    # detect faces
    Extract_Parts.part_croper(source_folder_path, faces_folder_path, Extract_Parts.Cascade_model.Face.value, True , 256, 256)
    # detect smiles
    Extract_Parts.part_croper(faces_folder_path, smiles_folder_path, Extract_Parts.Cascade_model.Smile.value, True, 64, 128)
    
    files = os.listdir(smiles_folder_path)
    genki4_labels, genki4_features = Read_labels()
    if not is_test:
        labels = []
    features = []
    for i in range(len(files)):
        image = cv2.imread(os.path.join(smiles_folder_path,files[i]))
        if not is_test:
            sample_index = Get_sample_index(files[i])

        # append genki4 features
        # feature_row = np.append(feature_row, genki4_features[i])
        feature_row = np.empty(0)

        # append HOG feature
        if get_HOG:
            fd = Get_HOG_singular(image,orientations, pixcels_per_cell)
            feature_row = np.append(feature_row,fd)

        # append LBP feature
        if get_LBP:
            lbp = Get_LBP_singular(image, P, R)
            feature_row = np.append(feature_row, lbp)

        # store feature and related label as a record
        features.append(feature_row)
        if not is_test:
            labels.append(genki4_labels[sample_index])
        print(files[i], " features extracted")

    if is_test:
        return features
    return features, labels

# train the model and return it
def Generate(source_folder_path, faces_folder_path, smiles_folder_path, calculate_accuracy = False, save_model = False):
    # get train x and y
    x_train, y_train = Get_labels_features(source_folder_path, faces_folder_path, smiles_folder_path, True, True , False)

    # split the data into training and testing sets (you can adjust the test_size parameter)
    if calculate_accuracy:
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    # create an SVM classifier
    clf = svm.SVC(kernel='linear')
    x_train = np.array(x_train)
    if calculate_accuracy:
        x_test = np.array(x_test)

    # train the classifier on the training data
    clf.fit(x_train, y_train)

    # make predictions on the test data
    if calculate_accuracy:
        y_pred = clf.predict(x_test)

    # calculate the accuracy of the classifier
    if calculate_accuracy:
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy * 100:.2f}%")

    # Save the trained model to a file
    if save_model:
        joblib.dump(clf, 'smile_detection.joblib')
    
    return clf

# load pre-trained model
def Load():
    return joblib.load('smile_detection.joblib')