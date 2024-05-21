import cv2 
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

train_data = np.load("train.npz")
test_data =np.load("test.npz")

x = train_data["x"]
y = train_data["y"]

x_test = test_data["x"]
y_test = test_data["y"]

print(x.shape)
print(y.shape)

inputs = keras.Input(shape=(28,28,1))

layers = keras.layers.Conv2D(filters = 32, kernel_size=(3,3), activation = "relu")(inputs)
layers = keras.layers.MaxPooling2D(pool_size = (2,2), strides=2)(layers)
layers = keras.layers.Conv2D(filters = 64, kernel_size=(3,3), activation = "relu")(layers)
layers = keras.layers.MaxPooling2D(pool_size = (2,2), strides=2)(layers)
layers = keras.layers.Conv2D(filters = 128, kernel_size=(3,3), activation = "relu")(layers)
layers = keras.layers.MaxPooling2D(pool_size = (2,2), strides=2)(layers)
layers = keras.layers.Flatten()(layers)

dense = keras.layers.Dense(units=64, activation = "relu")(layers)
dense = keras.layers.Dense(units=128, activation = "relu")(dense)
out = keras.layers.Dense(units=26, activation = "softmax")(dense)

model = keras.Model(inputs, out)

model.summary()

model.compile(optimizer = "adam", loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x=x, y=y, epochs=1,  validation_data = (x_test,y_test))

print("The validation accuracy is :", history.history['val_accuracy'])
print("The training accuracy is :", history.history['accuracy'])
print("The validation loss is :", history.history['val_loss'])
print("The training loss is :", history.history['loss'])

word_dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X', 24:'Y',25:'Z'}

pred = model.predict(x_test[:9])
print(x_test.shape)

fig, axes = plt.subplots(3,3, figsize=(8,9))
axes = axes.flatten()

for i,ax in enumerate(axes):
    img = np.reshape(x_test[i], (28,28))
    ax.imshow(img, cmap="Greys")

    pred = word_dict[np.argmax(y_test[i])]
    ax.set_title("Prediction: "+pred)
    ax.grid()

y=[]
for i in pred :
    y.append(np.argmax(i))
print("predicted values", y)

model.save("recog.h5")