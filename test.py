from os import listdir
import cv2
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import sys
import codecs
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
cap = cv2.VideoCapture(0)
# Dinh nghia class
class_name = ["bao"]
# load model da train
my_model = load_model("C:/Users/VICTUS/Documents/Zalo Received Files/Artifical intelligence/Artifical intelligence/model.keras")
my_model.load_weights("C:/Users/VICTUS/Documents/Zalo Received Files/Artifical intelligence/Artifical intelligence/data_train.keras")

while True:
    ret, image_org = cap.read()
    if not ret:
        continue
    image_org = cv2.resize(image_org, dsize=None, fx=0.5, fy=0.7)
    image = image_org.copy()
    image = cv2.resize(image, dsize=(128, 128))
    image = image.astype('float') * 1. / 255
    # Chuyen thanh tensor
    image = np.expand_dims(image, axis=0)
    # Predict
    predict = my_model.predict(image)
    print("This picture is: ", class_name[np.argmax(predict[0])], (predict[0]))
    #
    print("--")
    print(predict[0])
    print(np.argmax(predict[0]))
    print(class_name[np.argmax(predict[0])])

    print(np.max(predict[0], axis=0))
    if np.max(predict) >= 0.8:
        # show image
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 50)
        fontScale = 1.5
        color = (0, 255, 0)
        thickness = 2

        cv2.putText(image_org, class_name[np.argmax(predict)], org, font, fontScale, color, thickness, cv2.LINE_AA)

    cv2.imshow("Picture", image_org)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()