import cv2
from tflearn.layers.core import input_data, fully_connected, dropout
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.models.dnn import DNN
from tflearn.layers.estimator import regression

import numpy as np
cap = cv2.VideoCapture(0)

#make sure input_data 's placeholder shape corresponds to model shape
network = input_data(shape=[None, 28,28,1])
#shape: None (Placeholder for number of training sets in batch), 28 by 28 pixels, 1 channel (greyscale)
#now network becomes a 4D tensor with dimensions [batch, height, width, in_channels]
network = conv_2d(network, 16, [5,5], activation='relu')
#now network becomes a 4D tensor with dimensions [batch, new height, new width, n_filters]
network = max_pool_2d(network, [2,2])
#filter size [2,2], stride is implied: 2
network = conv_2d(network, 64, 3, activation='relu')
#use relu to account for non-linearity
network = max_pool_2d(network, 2)
#same as [2,2]
network = fully_connected(network, 512, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 24, activation='softmax')
network = regression(network, optimizer='sgd', loss='categorical_crossentropy',learning_rate=0.01)
#always remember to test different optimizers, set learning rates accordingly too
model = DNN(network, tensorboard_verbose=3, checkpoint_path='sign_language_model.ckpt')
model.load('ckpts/sign-language-classifier.tfl', weights_only=True)

while True:
    #ret stands for retrieve (cap.retrieve()), frame is the returned videcapture.grabbed
    #read combines features of retrieve and grab
    ret, frame = cap.read()

    # change to grayscale
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # resize frame before reshaping using numpy array
    frame = cv2.resize(frame, (28,28))
    cv2.imshow('object detect', frame)
    #reshape to input tensor dimensions
    gray = np.reshape(frame, (-1,28,28,1))
    print(gray)
    print(gray.shape)
    print(model.predict(gray))
    pred = np.argmax(model.predict(gray))

    if pred in range(9):
        print(chr(pred+65))
    elif pred in range(9,25):
        print(chr(pred+66))

    if cv2.waitKey(25) & 0xFF == ord('q'):
    #if we press q, exit window
        cv2.destroyAllWindows()
        break

# input = cv2.imread('B.jpg',0)
# input = cv2.resize(input, (28,28))
# cv2.imshow('image', input)
# gray = np.reshape(input, (-1,28,28,1))
# print(gray.shape)
# print(model.predict(gray))
# print(max(list(max(model.predict(gray)))))
# cv2.waitKey(0)
# cv2.destroyAllWindows()
