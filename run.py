import os
import time
import Model_Manager

# define folder paths
source_folder_path = 'genki4k'
faces_folder_path = os.path.join(source_folder_path,"faces")
smiles_folder_path = os.path.join(source_folder_path,"smiles")
test_folder_path = 'test'
test_face_folder_path = os.path.join(test_folder_path,"faces")
test_smile_folder_path = os.path.join(test_folder_path,"smiles")

# Record the start time
start_time = time.time()

# generate model
# model = Model_Manager.Generate(source_folder_path, faces_folder_path,smiles_folder_path, False, True)

# test
model = Model_Manager.Load()
x_test = Model_Manager.Get_labels_features(test_folder_path, test_face_folder_path, test_smile_folder_path, True, True, True)
prediction = model.predict(x_test)
print(prediction)

# record the end time
end_time = time.time()

# calculate the elapsed time
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")