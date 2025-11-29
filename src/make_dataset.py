import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

hugging_dataset = './hugging_face_dataset'

info_path= "./black-hole/GFI_database_with_labels/info/"
right_eye_path = "./black-hole/GFI_database_with_labels/Right_eye/"
left_eye_path = "./black-hole/GFI_database_with_labels/Left_eye/"


right_eye = np.loadtxt(f'{info_path}List_Right_eye.txt', dtype=str)
right_eye = pd.DataFrame(right_eye, columns=['filename', 'gender'])
right_eye['gender'] = pd.to_numeric(right_eye['gender'])
left_eye = np.loadtxt(f'{info_path}List_Left_eye.txt', dtype=str)
left_eye = pd.DataFrame(left_eye, columns=['filename','gender'])
left_eye['gender'] = pd.to_numeric(left_eye['gender'])

print(right_eye['filename'].head())
print(left_eye['filename'].head())

import glob, os
from pandas import Series
def get_image_paths(directory):
    return glob.glob(f"{directory}*.tiff")

right_eye_X = np.zeros(shape=(1500, 480, 640))
right_eye_y = np.zeros(shape=(1500))

for index, image_path in tqdm(enumerate(get_image_paths(right_eye_path))):
    right_eye_X[index] = plt.imread(image_path)
    filename = os.path.basename(image_path)
    label = right_eye.loc[right_eye['filename'] == filename, "gender"]
    right_eye_y[index] = label.values[0]

print("Read right eye")
left_eye_X = np.zeros(shape=(1500, 480, 640))
left_eye_y = np.zeros(shape=(1500))
for index, image_path in tqdm(enumerate(get_image_paths(left_eye_path))):
    image = plt.imread(image_path)
    mirrored_image = image[:, ::-1]
    left_eye_X[index] = mirrored_image
    filename = os.path.basename(image_path)
    label = left_eye.loc[left_eye['filename'] == filename, "gender"]
    left_eye_y[index] = label.values[0]

print(f"X shape: {right_eye_X.shape} and {left_eye_X.shape} ")
print(f"Y Shape: {right_eye_y.shape} and {left_eye_y.shape}")
right_eye_y_s = Series(right_eye_y)
left_eye_y_s = Series(left_eye_y)
print(right_eye_y_s.value_counts())
print(left_eye_y_s.value_counts())
plt.title(filename)
plt.imshow(right_eye_X[0])
plt.show()
plt.imshow(left_eye_X[0])
plt.show()


from sklearn.model_selection import train_test_split


total_x =np.concatenate([right_eye_X, left_eye_X])
total_y =np.concatenate([right_eye_y, left_eye_y])

# Determine the number of samples for each split
X_train, X_test, y_train, y_test = train_test_split(total_x, total_y, test_size=0.20, random_state=42) 
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

path = f"{os.environ['HOME']}/biometrics/black-hole/both-mirrored-eyes"

i = 0
for image, label in zip(X_train, y_train):
    if label == 0:
        plt.imsave(f'{path}/train/male/M_{i}.tiff', image)
    else:
        plt.imsave(f'{path}/train/female/F_{i}.tiff', image)
    i += 1

for image, label in zip(X_test, y_test):
    if label == 0:
        plt.imsave(f'{path}/test/male/M_{i}.tiff', image)
    else:
        plt.imsave(f'{path}/test/female/F_{i}.tiff', image)
    i += 1