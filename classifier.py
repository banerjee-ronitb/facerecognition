import cv2
import os
from PIL import Image
import numpy as np

def train_classifier(data_dir):
    dir_list = os.listdir(data_dir)
    dir_list.remove(".DS_Store")
    path = [os.path.join(data_dir, f) for f in dir_list]

    faces = []
    ids = []
    for image in path:
        img = Image.open(image).convert("L")
        imageNp = np.array(img, "uint8")
        id = int(os.path.split(image)[1].split(".")[1])
        faces.append(imageNp)
        ids.append(id)
    ids = np.array(ids)
    clf = cv2.face.LBPHFaceRecognizer().create()
    clf.train(faces, ids)
    clf.write("classifier.xml")
train_classifier("data")
