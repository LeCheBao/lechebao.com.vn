from os import listdir
import os
import cv2
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

# Thêm đoạn mã để đặt lại bảng mã tiêu chuẩn
import sys
import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

raw_folder = 'C:/Users/VICTUS/Documents/Zalo Received Files/Artifical intelligence/Artifical intelligence/data/'

def save_data(raw_folder=raw_folder):
    print("Bắt đầu xử lý ảnh...")
    pixels = []
    labels = []

    for folder in listdir(raw_folder):
        if folder != '.DS_Store':
            print('Folder= ', folder)
            for file in listdir(os.path.join(raw_folder, folder)):
                if file != '.DS_Store':
                    img_path = os.path.join(raw_folder, folder, file)
                    try:
                        img = cv2.imread(img_path)
                        if img is not None:
                            print('File= ', file)
                            resized_img = cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_AREA)
                            pixels.append(resized_img)
                            labels.append(folder)
                        else:
                            print(f'Failed to read image: {img_path}')
                    except Exception as e:
                        print(f'Error processing image {img_path}: {str(e)}')

    pixels = np.array(pixels)
    labels = np.array(labels)

    file = open('pix.data', 'wb')
    pickle.dump((pixels, labels), file)
    file.close()

def load_data():
    file = open('pix.data', 'rb')
    (pixels, labels) = pickle.load(file)
    file.close()
    print(pixels.shape)
    print(labels.shape)

    return pixels, labels

save_data()
X, Y = load_data()

# Chuyển đổi nhãn thành số nguyên
label_dict = {label: idx for idx, label in enumerate(np.unique(Y))}
Y = np.array([label_dict[label] for label in Y])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=100)

print(X_train.shape)
print(Y_train.shape)

def get_model():
    input_layer = Input(shape=(128, 128, 3), name='image_input')
    
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    output_layer = Dense(len(label_dict), activation='softmax', name='predictions')(x)

    my_model = Model(inputs=input_layer, outputs=output_layer)
    my_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return my_model

vggmodel = get_model()
filepath = 'C:/Users/VICTUS/Documents/Zalo Received Files/Artifical intelligence/Artifical intelligence/data_train.keras'
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

aug = ImageDataGenerator(rotation_range=20, zoom_range=0.1, rescale=1. / 255, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, brightness_range=[0.2, 1.5], fill_mode="nearest")
aug_val = ImageDataGenerator(rescale=1./255)

vgghist = vggmodel.fit(aug.flow(X_train, Y_train, batch_size=64), epochs=30, validation_data=aug_val.flow(X_test, Y_test, batch_size=64), callbacks=callbacks_list)
vggmodel.save('C:/Users/VICTUS/Documents/Zalo Received Files/Artifical intelligence/Artifical intelligence/model.keras')
