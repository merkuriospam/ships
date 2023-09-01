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

totalEpochs=1 # 50 original
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

# sns.countplot(x=train_df["category"].map(dictclass))
# plt.show()

# For a Convolutional Neural Network to function at its best, all the images provided to it must be of the same input size. 
# Hence we must first check if the images in our training set are of the same size.
path="./train/images/"
image_path1 = os.path.join(path,train_df["image"][0])
image_path2=os.path.join(path,train_df["image"][10])
print(plt.imread(image_path1).shape,plt.imread(image_path2).shape)

# As we can see here, the shape of two images from our dataset have different dimensions. 
# Hence it is important to resize the images to a fixed dimension to avoid irregularities.
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
# We have successfully reshaped the images into a fixed dimensionality of 128x128 pixels. 
# The refactor_size can be changed according to one's own preference. 
print(resized_image_list.shape)

# nrow=5
# ncol=4
# fig1 = plt.figure(figsize=(15,15))
# plt.suptitle('Before Resizing (Original)',size=32)
# for i in range (0,20):
#     plt.subplot(nrow,ncol,i+1)
#     plt.imshow(plt.imread(all_paths[i]))
#     plt.title('class = {x}, Ship is {y}'.format(x=train_df["category"][i],y=dictclass[train_df["category"][i]]))
#     plt.axis('Off')
#     plt.grid(False)

# fig2 = plt.figure(figsize=(15,15))
# fig2.suptitle('After Resizing',size=32)
# for i in range (0,20):
#     plt.subplot(nrow,ncol,i+1)
#     plt.imshow(resized_image_list[i])
#     plt.title('class = {x}, Ship is {y}'.format(x=train_df["category"][i],y=dictclass[train_df["category"][i]]))
#     plt.axis('Off')
#     plt.grid(False)
# plt.show()

# It is necessary for the neural network to be exposed to different variations of images, to improve accuracy and reduce overfitting. 
# Here the RandomFlip method flips the image according to the parameter given, and the RandomRotation method rotates the image.
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip('horizontal'),
  tf.keras.layers.RandomRotation(0.2),
])

class_values=train_df["category"]-1
print(class_values)

# Now let's start with making our model. Let's first make our class df and split the training data into train and test 
# to measure the accuracy(One can also think of this as a cross-validation set).
train_x,test_x,train_y,test_y = train_test_split(resized_image_list, class_values, train_size=0.70,test_size=0.30, random_state=1)
# We have successfully split the training data into a 7:3 ratio. 
print(train_x.shape,train_y.shape)

# We are going to use the tensorflow library to build our model. 
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

# Here I have given 3 convolution layers, with a maxpooling layer between each convolution layer. 
# The padding attribute adds padding to each convolved image ensuring that the output image has same dimensions as the input.
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='sigmoid'))
model.add(layers.Dense(5, activation='softmax'))
# The convolved images are then flattened into a 1-D array and are passed into the neural network with two layers. 
# The output layer is of 5 nodes as we have 5 classes.
model.summary()

# We the compile it using the loss function and an optimizer that helps the neural network to converge. 
# We test it on its accuracy to determine how well has our neural network performed.
model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
model.fit(train_x, train_y, epochs=totalEpochs, batch_size=128, shuffle=True)
# Our neural network has been trained. Let's test the accuracy of our NN on the cross-validation set.
# The accuracy of our model is not so great. I tried to tune the neural networks as best as I could and we have reached an accuracy of 69%. 
# We will use a very famous concept in Neural Networks called Transfer Learning
model.evaluate(test_x,test_y)

# Transfer Learning is a very important and widely used technique in the world of Machine Learning. 
# It has a very simple and basic concept - Taking information(in the case of NN, weights are the information) 
# learned from a task and aplying it on a task in-hand. In other words, we take the weights of a neural network 
# that has been trained on some other dataset and apply the weights to our neural network and voila, 
# we have a neural network with amazing accuracy in our hands, It is an amazing technique that brings all the machine learning 
# practioners like us together.
# For our transfer learning model, we will be using the weights of the Google Inception V3 Neural Network. 
# There are many choices, and any of them can be chosen, but I chose this due to its better results and robustness.
input_shape = (128, 128, 3)
# Here the include_top attribute is set to False to avoid the classification layers. 
# We will later add our own classification layer. The weights are taken from the imagenet trained neural network.
transfer_model = InceptionV3(input_shape=input_shape, include_top=False, weights=None)
transfer_model.load_weights('./inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')

# This does not allow the weights to change or to re-train. 
# After all we need the original weights from Inception for better accuracy.
transfer_model.trainable=False
transfer_model.output
transfer_model.summary()
# Inception basically consists of many layers and each group of layers has a name like mixed0, mixed5 etc till mixed10. 
# You can the train the neural network on the entire Inception NN or can choose a subset of layers by testing them out. 
# I had the best accuracy till the mixed4 layer so I would be continuing with this.
transfer_final_layer = transfer_model.get_layer('mixed4')

# Now, we can add our own dense layers before we add the classification layer or directly add the classification layer, 
# whichever gives a better result. I have directly added a classification layer and hence used it to predict.
x = data_augmentation(transfer_final_layer.output)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(5, activation='softmax')(x)

transfer_model.input
transfer_model = tf.keras.Model(transfer_model.input, x)
transfer_model.summary()

# Here the callback function refers to the function that would stop the training of the neural network if the loss function 
# does not reduce after a certain specified number of epochs. That specified number is known as the patience.
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=4)
transfer_model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001),
             metrics = ['accuracy'],
             loss='sparse_categorical_crossentropy')

transfer_model.fit(train_x, train_y,validation_steps=50, verbose=1, epochs=totalEpochs, callbacks=[callback])
transfer_model.evaluate(test_x,test_y)

test_df = pd.read_csv("./test_ApKoW4T.csv")
print(test_df.head())

# Before we predict, we must first preprocess the data to resize the images.
resized_test_images=[]
refactor_size=128
for i in range(test_df.shape[0]):
    image_path=os.path.join(path,test_df["image"][i])
    img=tf.keras.utils.load_img(image_path,target_size=(refactor_size,refactor_size))
    img_vals = tf.image.convert_image_dtype(img, tf.float32)
    imgarr = tf.keras.utils.img_to_array(img_vals)
    resized_test_images.append(imgarr)
resized_test_images = np.asarray(resized_test_images)

# fig2 = plt.figure(figsize=(15,15))
# fig2.suptitle('Test Images',size=32)
# for i in range (0,20):
#     plt.subplot(5,4,i+1)
#     plt.imshow(resized_test_images[i])
#     plt.axis('Off')
#     plt.grid(False)

print(resized_test_images.shape)

# Let's predict
pred=transfer_model.predict(resized_test_images)
sub = pd.read_csv('./sample_submission_ns2btKE.csv')
sub['category'] = np.argmax(pred, axis=1) + 1 
print(sub.head(20))

sub.to_csv('submissionMartinBorches.csv', index=False)

# We can improve our model or at the very least find the inaccuracies made by our model by getting hold of the Missclassified images.
predMiss = transfer_model.predict(test_x)
pred_label = np.argmax(predMiss,axis=1) 
actual_label = test_y 
pred_label
labels = ["Cargo","Military","Carrier","Cruise","Tankers"]
sns.heatmap(confusion_matrix(actual_label,pred_label),annot=True,fmt='g',xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix', fontsize = 20) 
plt.xlabel('Predict', fontsize = 15) 
plt.ylabel('Actual', fontsize = 15) 
plt.show()