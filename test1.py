################
# benign = 0 ; Cancer = 1; Normal = 2
#
#
#################################
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from urllib import urlretrieve
import cPickle as pickle
import os
import gzip
import numpy as np
from os import listdir
from os.path import isfile, join
import glob
import theano
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
# from nolearn.lasagne import visualize
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from skimage import io
import theano.tensor as T
from nolearn.lasagne import BatchIterator

try:
    from lasagne.layers.cuda_convnet import Conv2DCCLayer as Conv2DLayer
    from lasagne.layers.cuda_convnet import MaxPool2DCCLayer as MaxPool2DLayer
except ImportError:
    Conv2DLayer = layers.Conv2DLayer
    MaxPool2DLayer = layers.MaxPool2DLayer

# Convert integer to 32 floatpoint value
def float32(k):
    return np.cast['float32'](k)

# Create CNN
net1 = NeuralNet(
    layers=[('input', layers.InputLayer),
            ('conv2d1', layers.Conv2DLayer),
            ('maxpool1', layers.MaxPool2DLayer),
            ('conv2d2', layers.Conv2DLayer),
            ('maxpool2', layers.MaxPool2DLayer),
            ('conv2d3', layers.Conv2DLayer),
            ('maxpool3', layers.MaxPool2DLayer),
            ('dropout1', layers.DropoutLayer),
            ('dense', layers.DenseLayer),
            ('dropout2', layers.DropoutLayer),
            ('output', layers.DenseLayer),
            ],
    # input layer
    input_shape=(None, 1, 256, 256),
    # layer conv2d1
    conv2d1_num_filters=64,
    conv2d1_filter_size=(3, 3),
    conv2d1_nonlinearity=lasagne.nonlinearities.rectify,
    conv2d1_W=lasagne.init.GlorotUniform(),
    # layer maxpool1
    maxpool1_pool_size=(2, 2),
    # layer conv2d2
    conv2d2_num_filters=128,
    conv2d2_filter_size=(3, 3),
    conv2d2_nonlinearity=lasagne.nonlinearities.rectify,
    # layer maxpool2
    maxpool2_pool_size=(2, 2),
    # layer conv2d2
    conv2d3_num_filters=256,
    conv2d3_filter_size=(3, 3),
    conv2d3_nonlinearity=lasagne.nonlinearities.rectify,
    # layer maxpool2
    maxpool3_pool_size=(2, 2),
    # dropout1
    dropout1_p=0.5,
    # dense
    dense_num_units=500,
    dense_nonlinearity=lasagne.nonlinearities.rectify,
    # dropout2
    dropout2_p=0.5,
    # output
    output_nonlinearity=lasagne.nonlinearities.softmax,
    output_num_units=3,
    # optimization method params
    update=nesterov_momentum,
    update_learning_rate=theano.shared(float32(0.01)),
    update_momentum=theano.shared(float32(0.9)),
    max_epochs=25,
    verbose=1,
    )

X = []
y=[]
X_benign=[]
y_benign=[]

X_benign_test=[]
y_benign_test=[]
# load training and testing data
filename_benign_train = ["/home/ubuntu/data/ROIs/benigns/%d.png" % r for r in range(0, 100)]
filename_benign_test = ["/home/ubuntu/data/ROIs/benigns/%d.png" % r for r in range(70, 103)]
filename_cancers = ["/home/ubuntu/data/ROIs/cancers/%d.png" % r for r in range(0, 178)]

for fn in filename_benign_train:
    im = io.imread(str(fn),True)
    im1=im.reshape(-1, 1, 256, 256)
    X_benign.append(im1-0.5)
    y_benign.append(0)

for fn in filename_benign_test:
    im = io.imread(str(fn),True)
    im1=im.reshape(-1, 1, 256, 256)
    X_benign_test.append(im1-0.5)
    y_benign_test.append(0)

for fn in filename_cancers:
    im = io.imread(str(fn),True)
    im1=im.reshape(-1, 1, 256, 256)
    X.append(im1-0.5)
    y.append(1)

mypath='/home/ubuntu/data/ROIs/normals/'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
for n in range(len(onlyfiles)):
  images = io.imread(join(mypath,onlyfiles[n]), True)
  im1 = images.reshape(-1, 1, 256, 256)
  X.append(im1-0.5)
  y.append(2)


print y


# Random spliting the data(Normals and Cancers)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print y_train
print len(y_train)

# Insert benign images into training dataset
for i in range(0,100):
    if i*6 <= 560:
        X_train.insert((i*6),X_benign[i])
        y_train.insert((i*6),y_benign[i])
    elif i*6 > 560:
        X_train.insert((i*3),X_benign[i])
        y_train.insert((i*3),y_benign[i])

# Delete the last 5 items in the dataset. 
# The total number of training images is 550.
del X_train[8:13]
del y_train[8:13]
# Convert training data into 32bit floatpoint
X_train = np.array(X_train).astype(np.float32)
y_train = np.array(y_train).astype(np.int32)
X_train = X_train.reshape(-1,1,256,256)

print y_train
print len(y_train)
# Add benign images into testing dataset
for i in range(0,23):
    X_test.append(X_benign_test[i])
    y_test.append(y_benign_test[i])

X_test = np.array(X_test).astype(np.float32)
y_test = np.array(y_test).astype(np.int32)
X_test = X_test.reshape(-1,1,256,256)

# Create batch iterator to avoid GPU memory overhead
from nolearn.lasagne import BatchIterator
bi = BatchIterator(batch_size=50)
for Xb, yb in bi(X_train, y_train):
    nn = net1.fit(Xb,yb)

import cPickle as pickle
with open('net1.pickle', 'wb') as f:
    pickle.dump(net1, f, -1)


preds = net1.predict(X_test)
cm = confusion_matrix(y_test, preds)
plt.matshow(cm)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('confusion_mean.png')
plt.close()

from sklearn.metrics import classification_report, accuracy_score

print(classification_report(y_test, preds)) # Classification on each digit




train_loss = np.array([i["train_loss"] for i in net1.train_history_])
valid_loss = np.array([i["valid_loss"] for i in net1.train_history_])
plt.plot(train_loss, linewidth=3, label="train")
plt.plot(valid_loss, linewidth=3, label="valid")
plt.grid()
plt.legend()
plt.xlabel("epoch")
plt.ylabel("loss")
plt.savefig('lose.png')
plt.close()

