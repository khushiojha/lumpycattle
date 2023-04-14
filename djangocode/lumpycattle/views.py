
from django.http import HttpResponse
from django.shortcuts import render,redirect
import cv2
import base64
from joblib import load
from django.core.files.storage import FileSystemStorage
from django.core.files.storage import default_storage
def extract_features(image):
    img_size = (64, 64)
    # Resize the image to the desired size
    resized_image = cv2.resize(image, img_size)
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    # Flatten the image to a 1D array
    flattened_image = gray_image.flatten()
    return flattened_image

svm_classifier = load('static/model.joblib')

def predict_image(img_path):
    test_image = img_path
    image_read = cv2.imread(test_image)
    results = svm_classifier.predict([extract_features(image_read)])
    return results

def index(request):
    return render(request,"index.html")

def upload(request):
    if request.method == "POST":
        uploaded_file = request.FILES['file']
        fs = FileSystemStorage()
        filename = fs.save(uploaded_file.name,uploaded_file)
        uploaded_file_path = fs.path(filename)
        image = cv2.imread(uploaded_file_path)
        results = svm_classifier.predict([extract_features(image)])
        with open(uploaded_file_path, 'rb') as f:
            img_data = f.read()
        img_base64 = base64.b64encode(img_data).decode('utf-8')
        ans = results[0]
        if ans==0:
            result = "MILD"
        elif ans==1:
            result = "Normal"
        else:
            result = "Severe"
    context = {'message': 'Image uploaded successfully.', 'uploaded_img': img_base64, 'type':result}
    return render(request,"results.html",context)