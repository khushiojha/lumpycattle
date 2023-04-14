import os
import cv2
import glob
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Define the folders containing the images of the 3 different classes
folder1 = './DATASET/MILD/'
folder2 = './DATASET/normal cattle/'
folder3 = './DATASET/severe/'

# Define the size of the images
img_size = (64, 64)

# Define the feature extractor function
def extract_features(image):
    # Resize the image to the desired size
    resized_image = cv2.resize(image, img_size)
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    # Flatten the image to a 1D array
    flattened_image = gray_image.flatten()
    return flattened_image

# Load the images and extract the features
X = []
y = []

# Load the images from folder1 and assign class label 0
for name in glob.glob('./DATASET/MILD/*'):
    # print(name)
    try:
        image = cv2.imread(name)
        features = extract_features(image)
        X.append(features)
        y.append(0)
    except:
        print("Got an error.. and moving Forward")
        continue

# Load the images from folder2 and assign class label 1
for name in glob.glob('./DATASET/normal cattle/*'):
    # print(name)
    try:
        image = cv2.imread(name)
        features = extract_features(image)
        X.append(features)
        y.append(1)
    except:
        print("Got an error.. and moving Forward")
        continue

# Load the images from folder3 and assign class label 2
for name in glob.glob('./DATASET/severe/*'):
    # print(name)
    try:
        image = cv2.imread(name)
        features = extract_features(image)
        X.append(features)
        y.append(0)
    except:
        print("Got an error.. and moving Forward")
        continue

# Convert the data to NumPy arrays
X = np.array(X)
y = np.array(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the SVM classifier
svm_classifier = svm.SVC(kernel='rbf', C=5, gamma='scale',decision_function_shape='ovo')

# Train the SVM classifier on the training set
svm_classifier.fit(X_train, y_train)

# Test the SVM classifier on the testing set
y_pred = svm_classifier.predict(X_test)

# To Predict the new Image
test_image = './DATASET/normal cattle/download.jpg'
image_read = cv2.imread(test_image)
print("Testing for Image: ",svm_classifier.predict([extract_features(image_read)]))

# Print the classification report

print("----CLASSIFICATION REPORT------")
print("Accuracy Score : ",round(accuracy_score(y_test,y_pred)*100,2),'%')
print(classification_report(y_test, y_pred))
