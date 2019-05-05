# -*- coding: utf-8 -*-
###############################################################################
#this code helps to train a sequential custom model on the dataset of your interest and 
#visualize the confusion matrix, ROC and AUC curves
###############################################################################
#load the libraries
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from CNN.resnet import ResNet
from CNN import config
from keras.callbacks import LearningRateScheduler
from keras.layers import add
from keras.regularizers import l2
from keras import backend as K
import numpy as np
import time
from keras.models import Sequential
from keras.layers import Conv2D, Activation, Dense, MaxPooling2D, Flatten, Dropout
from sklearn.metrics import log_loss
from keras.optimizers import SGD
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from keras.utils import np_utils
#from CNN import config
from evaluation import plot_confusion_matrix
from sklearn.metrics import average_precision_score
from load_data import load_resized_training_data, load_resized_validation_data
#########################image characteristics#################################
img_rows=64 #dimensions of image
img_cols=64
channel = 3 #RGB
num_classes = 2 
batch_size = 16
num_epoch = 3
from imutils import paths
import argparse
######################################################################
######################################################
import matplotlib
matplotlib.use("Agg")
 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--plot", type=str, default="ROC.png",
	help="path to output True Positive Rate plot")
ap.add_argument("-p", "--plot", type=str, default="confusion.png",
	help="path to output  plot")
ap.add_argument("-p", "--plot", type=str, default="plot1.png",
	help="path to output loss plot")
ap.add_argument("-p", "--plot", type=str, default="plot2.png",
	help="path to output accuracy plot")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
args = vars(ap.parse_args())






"""configuring the customized model"""
"""
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(100,100,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

#fix the optimizer
sgd = SGD(lr=0.000001, decay=1e-6, momentum=0.9, nesterov=True) #try varying this for your task and see the best fit

#compile the model
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

###############################################################################
#load data
#X_train, Y_train = load_training_data()
X_train, Y_train = load_resized_training_data(img_rows, img_cols)
X_val, Y_val = load_resized_validation_data(img_rows, img_cols)
#print the shape of the data
print(X_train.shape, Y_train.shape)
###############################################################################
#train the model
t=time.time()
"""



# define the # of epochs, initial learning rate and batch size
num_epochs = num_epoch
init_lr= 1e-1
bs = 15
# create a function called polynomial decay which helps us decay our 
# learning rate after each epoch

def poly_decay(epoch):
	# initialize the maximum # of epochs, base learning rate,
	# and power of the polynomial
	maxEpochs = num_epochs
	baseLR = init_lr
	power = 1.0  # turns our polynomial decay into a linear decay
 
	# compute the new learning rate based on polynomial decay
	alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power
 
	# return the new learning rate
	return alpha

# determine the # of image paths in training/validation/testing directories
totalTrain = len(list(paths.list_images(config.train_path)))
totalVal = len(list(paths.list_images(config.val_path)))
totalTest = len(list(paths.list_images(config.test_path)))

# initialize the training data augmentation object
# randomly shifts, translats, and flips each training sample
trainAug = ImageDataGenerator(
	rescale=1 / 255.0,
	rotation_range=20,
	zoom_range=0.05,
	width_shift_range=0.05,
	height_shift_range=0.05,
	shear_range=0.05,
	horizontal_flip=True,
	fill_mode="nearest")
 
# initialize the validation (and testing) data augmentation object
valAug = ImageDataGenerator(rescale=1 / 255.0)

# initialize the training generator
trainGen = trainAug.flow_from_directory(
	config.train_path,
	class_mode="categorical",
	target_size=(64, 64),
	color_mode="rgb",
	shuffle=True,
	batch_size=bs)
 
# initialize the validation generator
valGen = valAug.flow_from_directory(
	config.val_path,
	class_mode="categorical",
	target_size=(64, 64),
	color_mode="rgb",
	shuffle=False,
	batch_size=bs)
 
# initialize the testing generator
testGen = valAug.flow_from_directory(
	config.test_path,
	class_mode="categorical",
	target_size=(64, 64),
	color_mode="rgb",
	shuffle=False,
	batch_size=bs)

# initialize our ResNet model and compile it
model = ResNet.build(64, 64, 3, 2, (3, 4, 6),
	(64, 128, 256, 512), reg=0.0005)
opt = SGD(lr=init_lr, momentum=0.9)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# define our set of callbacks and fit the model
callbacks = [LearningRateScheduler(poly_decay)]
hist = model.fit_generator(
	trainGen,
	steps_per_epoch=totalTrain // bs,
	validation_data=valGen,
	validation_steps=totalVal // bs,
	epochs=num_epochs,
	callbacks=callbacks)


"""


hist = model.fit(X_train, Y_train, batch_size=batch_size, epochs=num_epoch, verbose=1, 
                    shuffle=True, validation_data=(X_val,Y_val))
"""

#####################################
###############################################################################
#predict on the validation data
X_test, Y_test = load_resized_validation_data(img_rows, img_cols)
print(X_test.shape, Y_test.shape)

# Make predictions
print('-'*30)
print('Predicting on the validation data...')
print('-'*30)
y_pred = model.predict(X_test, batch_size=batch_size, verbose=1)

# compute the accuracy
Test_accuracy = accuracy_score(Y_test.argmax(axis=-1),y_pred.argmax(axis=-1))
print("Test_Accuracy = ",Test_accuracy)

#compute the ROC-AUC values
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(Y_test.ravel(), y_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

#Plot ROC curve for the positive class
plt.figure(figsize=(20,10), dpi=300)
lw = 1 #true class label
plt.plot(fpr[1], tpr[1], color='red',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristics')
plt.legend(loc="lower right")
plt.savefig(args["ROC"])


## computhe the cross-entropy loss score
score = log_loss(Y_test,y_pred)
print(score)

## compute the average precision score
prec_score = average_precision_score(Y_test,y_pred)  
print(prec_score)

# transfer it back
y_pred = np.argmax(y_pred, axis=1)
Y_test = np.argmax(Y_test, axis=1)
print(y_pred)
print(Y_test)

np.savetxt('custom_model_y_pred.csv',y_pred,fmt='%i',delimiter = ",")
np.savetxt('custom_model_Y_test.csv',Y_test,fmt='%i',delimiter = ",")
#################compute confusion matrix######################################

#plot the confusion matrix
target_names = ['class 0(abnormal)', 'class 1(normal)']
print(classification_report(Y_test,y_pred,target_names=target_names))
print(confusion_matrix(Y_test,y_pred))
cnf_matrix = (confusion_matrix(Y_test,y_pred))
np.set_printoptions(precision=4)
plt.figure(figsize=(20,10), dpi=300)
# Plot non-normalized confusion matrix
plot_confusion_matrix(cnf_matrix, classes=target_names,
                  title='Confusion matrix')
plt.savefig(args["confusion"])

###############################################################################
# visualizing losses and accuracy
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(num_epoch)

plt.figure(1,figsize=(8,4), dpi=100)
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
plt.style.use(['classic'])
plt.savefig(args["plot1"])


plt.figure(2,figsize=(8,4), dpi=100)
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'])
plt.style.use(['classic'])
plt.savefig(args["plot2"])

###############################################################################


N = num_epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), hist.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), hist.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), hist.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), hist.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])

model_json = model.to_json()
with open("mdm.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("mdm.h5")
