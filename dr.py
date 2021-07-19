import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time

labels = os.listdir("C:/Users/mvsub/OneDrive/Documents/ML Projects/Drowsiness dataset/dataset_new/train")

labels
import matplotlib.pyplot as plt
plt.imshow(plt.imread("C:/Users/mvsub/OneDrive/Documents/ML Projects/Drowsiness dataset/dataset_new/train/Closed/_0.jpg"))

a=plt.imread("C:/Users/mvsub/OneDrive/Documents/ML Projects/Drowsiness dataset/dataset_new/train/yawn/1.jpg")
a.shape

plt.imshow(a)

def face_for_yawn(direc="C:/Users/mvsub/OneDrive/Documents/ML Projects/Drowsiness dataset/dataset_new/train", face_cas_path="C:/Users/mvsub/OneDrive/Documents/ML Projects/Drowsiness dataset/dataset_new/haar-cascade-files-master/haarcascade_frontalface_default.xml"):
    yaw_no = []
    IMG_SIZE = 145
    categories = ["yawn", "no_yawn"]
    for category in categories:
        path_link = os.path.join(direc, category)
        class_num1 = categories.index(category)
        print(class_num1)
        for image in os.listdir(path_link):
            image_array = cv2.imread(os.path.join(path_link, image), cv2.IMREAD_COLOR)
            face_cascade = cv2.CascadeClassifier(face_cas_path)
            faces = face_cascade.detectMultiScale(image_array, 1.3, 5)
            for (x, y, w, h) in faces:
                img = cv2.rectangle(image_array, (x, y), (x+w, y+h), (0, 255, 0), 2)
                roi_color = img[y:y+h, x:x+w]
                resized_array = cv2.resize(roi_color, (IMG_SIZE, IMG_SIZE))
                yaw_no.append([resized_array, class_num1])
    return yaw_no


yawn_no_yawn = face_for_yawn()

def get_data(dir_path="C:/Users/mvsub/OneDrive/Documents/ML Projects/Drowsiness dataset/dataset_new/train", face_cas="C:/Users/mvsub/OneDrive/Documents/ML Projects/Drowsiness dataset/dataset_new/haar-cascade-files-master/haarcascade_frontalface_default.xml", eye_cas="C:/Users/mvsub/OneDrive/Documents/ML Projects/Drowsiness dataset/dataset_new/haar-cascade-files-master"):
    labels = ['Closed', 'Open']
    IMG_SIZE = 145
    data = []
    for label in labels:
        path = os.path.join(dir_path, label)
        class_num = labels.index(label)
        class_num +=2
        print(class_num)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
                resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                data.append([resized_array, class_num])
            except Exception as e:
                print(e)
    return data
data_train = get_data()

def append_data():
#     total_data = []
    yaw_no = face_for_yawn()
    data = get_data()
    yaw_no.extend(data)
    return np.array(yaw_no)
new_data = append_data()

X = []
y = []
for feature, label in new_data:
    X.append(feature)
    y.append(label)

X = np.array(X)
X = X.reshape(-1, 145, 145, 3)

from sklearn.preprocessing import LabelBinarizer
label_bin = LabelBinarizer()
y = label_bin.fit_transform(y)
#label array
y = np.array(y)
#train test split
from sklearn.model_selection import train_test_split
seed = 42
test_size = 0.30
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, test_size=test_size)
#length of X_test
len(X_test)
len(X_train)
len(y_train)
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf


train_generator = ImageDataGenerator(rescale=1/255, zoom_range=0.2, horizontal_flip=True, rotation_range=30)
test_generator = ImageDataGenerator(rescale=1/255)

train_generator = train_generator.flow(np.array(X_train), y_train, shuffle=False)
test_generator = test_generator.flow(np.array(X_test), y_test, shuffle=False)

model = Sequential()

model.add(Conv2D(256, (3, 3), activation="relu", input_shape=X_train.shape[1:]))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(MaxPooling2D(2, 2))

model.add(Flatten())
model.add(Dropout(0.5))

model.add(Dense(64, activation="relu"))
model.add(Dense(4, activation="softmax"))

model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer="adam")

model.summary()
history = model.fit(train_generator,batch_size=32, epochs=32, validation_data=test_generator, shuffle=True, validation_steps=len(test_generator))

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(accuracy))

plt.plot(epochs, accuracy, "b", label="trainning accuracy")
plt.plot(epochs, val_accuracy, "r", label="validation accuracy")
plt.legend()
plt.show()

plt.plot(epochs, loss, "b", label="trainning loss")
plt.plot(epochs, val_loss, "r", label="validation loss")
plt.legend()  
plt.show()

model.save("drowiness_new6.h5")
model.save("drowiness_new6.model")
#Prediction
prediction = model.predict_classes(X_test)
prediction

labels_new = ["yawn", "no_yawn", "Closed", "Open"]

from sklearn.metrics import classification_report
print(classification_report(np.argmax(y_test, axis=1), prediction, target_names=labels_new))

labels_new = ["yawn", "no_yawn", "Closed", "Open"]
IMG_SIZE = 145
def prepare(filepath, face_cas="C:/Users/mvsub/OneDrive/Documents/ML Projects/Drowsiness dataset/dataset_new/haar-cascade-files-master/haarcascade_frontalface_default.xml"):
    img_array = cv2.imread(filepath, cv2.IMREAD_COLOR)
    img_array = img_array / 255
    resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return resized_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)

model = tf.keras.models.load_model("C:/Users/mvsub/OneDrive/Documents/ML Projects/Drowsiness dataset/drowiness_new6.h5")


# prepare("../input/drowsiness-dataset/train/no_yawn/1068.jpg")
prediction = model.predict([prepare("C:/Users/mvsub/OneDrive/Documents/ML Projects/Drowsiness dataset/dataset_new/test/yawn/14.jpg")])
np.argmax(prediction)
