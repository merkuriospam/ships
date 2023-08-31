import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras import datasets, layers, models
from tensorflow.math import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.inception_v3 import InceptionV3
import warnings
import os
warnings.filterwarnings('ignore')

totalEpochs=5 # 50 original
train_df=pd.read_csv('./train/train.csv')
print(train_df.head())
print(train_df.shape)

dictclass = {
    1: 'Cargo',
    2: 'Military',
    3: 'Carrier',
    4: 'Cruise',
    5: 'Tankers'
}

sns.countplot(x=train_df["category"].map(dictclass))
# plt.show()

path="./train/images/"
image_path1 = os.path.join(path,train_df["image"][0])
image_path2=os.path.join(path,train_df["image"][10])
print(plt.imread(image_path1).shape,plt.imread(image_path2).shape)

resized_image_list=[]
all_paths=[]
refactor_size=128
for i in range(train_df.shape[0]):
    image_path=os.path.join(path,train_df["image"][i])
    img=tf.keras.utils.load_img(image_path,target_size=(refactor_size,refactor_size))
    img_vals = tf.image.convert_image_dtype(img, tf.float32)
    imgarr = tf.keras.utils.img_to_array(img_vals)
    
    resized_image_list.append(imgarr)
    all_paths.append(image_path)
resized_image_list = np.asarray(resized_image_list)
print(resized_image_list.shape)

nrow=5
ncol=4
fig1 = plt.figure(figsize=(15,15))
plt.suptitle('Before Resizing (Original)',size=32)
for i in range (0,20):
    plt.subplot(nrow,ncol,i+1)
    plt.imshow(plt.imread(all_paths[i]))
    plt.title('class = {x}, Ship is {y}'.format(x=train_df["category"][i],y=dictclass[train_df["category"][i]]))
    plt.axis('Off')
    plt.grid(False)

fig2 = plt.figure(figsize=(15,15))
fig2.suptitle('After Resizing',size=32)
for i in range (0,20):
    plt.subplot(nrow,ncol,i+1)
    plt.imshow(resized_image_list[i])
    plt.title('class = {x}, Ship is {y}'.format(x=train_df["category"][i],y=dictclass[train_df["category"][i]]))
    plt.axis('Off')
    plt.grid(False)
# plt.show()

data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip('horizontal'),
  tf.keras.layers.RandomRotation(0.2),
])

class_values=train_df["category"]-1
print(class_values)

train_x,test_x,train_y,test_y = train_test_split(resized_image_list, class_values, train_size=0.70,test_size=0.30, random_state=1)
print(train_x.shape,train_y.shape)

model = tf.keras.models.Sequential()
model.add(tf.keras.Input(shape=(128,128,3)))
model.add(data_augmentation)
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((3, 3),strides=2))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((3, 3),strides=2))
model.add(layers.Conv2D(16, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((3, 3),strides=2))
model.summary()

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='sigmoid'))
model.add(layers.Dense(5, activation='softmax'))
model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
model.fit(train_x, train_y, epochs=totalEpochs, batch_size=128, shuffle=True)
model.evaluate(test_x,test_y)

input_shape = (128, 128, 3)
transfer_model = InceptionV3(input_shape=input_shape, include_top=False, weights=None)
transfer_model.load_weights('./inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')

transfer_model.trainable=False
transfer_model.output
transfer_model.summary()
transfer_final_layer = transfer_model.get_layer('mixed4')

x = data_augmentation(transfer_final_layer.output)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(5, activation='softmax')(x)

transfer_model.input
transfer_model = tf.keras.Model(transfer_model.input, x)
transfer_model.summary()

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=4)
transfer_model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001),
             metrics = ['accuracy'],
             loss='sparse_categorical_crossentropy')

transfer_model.fit(train_x, train_y,validation_steps=50, verbose=1, epochs=totalEpochs, callbacks=[callback])
transfer_model.evaluate(test_x,test_y)

test_df = pd.read_csv("./test_ApKoW4T.csv")
print(test_df.head())

resized_test_images=[]
refactor_size=128
for i in range(test_df.shape[0]):
    image_path=os.path.join(path,test_df["image"][i])
    img=tf.keras.utils.load_img(image_path,target_size=(refactor_size,refactor_size))
    img_vals = tf.image.convert_image_dtype(img, tf.float32)
    imgarr = tf.keras.utils.img_to_array(img_vals)
    resized_test_images.append(imgarr)
resized_test_images = np.asarray(resized_test_images)

fig2 = plt.figure(figsize=(15,15))
fig2.suptitle('Test Images',size=32)
for i in range (0,20):
    plt.subplot(5,4,i+1)
    plt.imshow(resized_test_images[i])
    plt.axis('Off')
    plt.grid(False)

print(resized_test_images.shape)

pred=transfer_model.predict(resized_test_images)
sub = pd.read_csv('./sample_submission_ns2btKE.csv')
sub['category'] = np.argmax(pred, axis=1) + 1 
print(sub.head(20))

sub.to_csv('submissionMartinBorches.csv', index=False)