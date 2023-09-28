import cv2
import os
from enum import Enum
from skimage.feature import hog, local_binary_pattern
import numpy as np
from sklearn import svm
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Record the start time
start_time = time.time()

# define folder paths
source_folder_path = 'genki4k'
faces_folder_path = 'faces'
smiles_folder_path = 'smiles'

# define opencv cascade pre-trained models
class Cascade_model(Enum):
    class Face(Enum):
        Name = "face"
        Model = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        Scale_factor = 1.1
        Min_Neighbors = 15
    class Smile(Enum):
        Name = "smile"
        Model = cv2.CascadeClassifier('haarcascade_smile.xml')
        Scale_factor = 1.8
        Min_Neighbors = 40

# get best size based on aspect ratio
def get_size_ratio(min_width, min_height, ratio = (4,3)):
    if ratio[0] > ratio[1]:
        if min_width > min_height:
            min_height = min_width * ratio[1] / ratio[0]
        else:
            min_width = min_height * ratio[0] / ratio[1]
    else:
        if min_width > min_height:
            min_height = min_width * ratio[1] / ratio[0]
        else:
            min_width = min_height * ratio[0] / ratio[1]
    return round(min_width), round(min_height)

# crop parts from images and save in another folder
def part_croper(input_folder_path, target_folder_path, cascade_model, should_resize = False, height = 128, width = 128):
    images = os.listdir(input_folder_path)

    # remove old files
    files_to_remove = os.listdir(target_folder_path)
    if len(files_to_remove) > 0:
        for file in files_to_remove:
            os.remove(os.path.join(target_folder_path,file))

    if len(images) > 0:
        for image in images:
            img = cv2.imread(os.path.join(input_folder_path,image))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # detects parts in the input image
            parts = cascade_model.Model.value.detectMultiScale(gray,
            cascade_model.Scale_factor.value,
            cascade_model.Min_Neighbors.value)

            # If it doesn't exist, create the target folder
            if not os.path.exists(target_folder_path):
                os.makedirs(target_folder_path)

            # save detected parts
            if len(parts) > 0:
                for i, (x, y, w, h) in enumerate(parts):

                    # get the best width and height for being ready to resize
                    if should_resize:
                        w, h = get_size_ratio(w,h, (width, height))
                    part = img[y:y + h, x:x + w]
                    file_name = os.path.join(target_folder_path, f'{image[:-4]}-{cascade_model.Name.value}{i+1}.jpg')
                    if os.path.exists(file_name):
                        os.remove(file_name)

                    # resize the photo for consistency
                    if should_resize:
                        part = cv2.resize(part, (width, height), interpolation=cv2.INTER_LINEAR)
                    cv2.imwrite(file_name, part)
                    print(file_name)
            else :
                print(image,f"No {cascade_model.Name.value} detected!")

# detect faces
part_croper(source_folder_path, faces_folder_path, Cascade_model.Face.value, True , 256, 256)

# detect smiles
part_croper(faces_folder_path, smiles_folder_path, Cascade_model.Smile.value, 128, 64)

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

def Get_labels_features(input_folder_path, get_HOG, get_LBP, orientations = 8, pixcels_per_cell = 4, P = 1, R = 8):
    files = os.listdir(input_folder_path)
    counter = 0
    genki4_labels, genki4_features = Read_labels()
    labels = []
    features = []
    for i in range(len(files)):
        image = cv2.imread(os.path.join(input_folder_path,files[i]))
        sample_index = Get_sample_index(files[i])

        # append genki4 features
        feature_row = np.empty(0)
        feature_row = np.append(feature_row, genki4_features[i])

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
        labels.append(genki4_labels[sample_index])
        counter = counter + 1
        print(files[i], " features extracted")
    return features, labels, counter

# get train x and y
x_train, y_train, counter = Get_labels_features(smiles_folder_path, True, True, 8, 4, 1, 8)

# split the data into training and testing sets (you can adjust the test_size parameter)
X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# create an SVM classifier
clf = svm.SVC(kernel='linear')
X_train = np.array(X_train)
X_test = np.array(X_test)

# train the classifier on the training data
clf.fit(X_train, y_train)

# make predictions on the test data
y_pred = clf.predict(X_test)

# calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(counter, "samples")

# record the end time
end_time = time.time()

# calculate the elapsed time
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")