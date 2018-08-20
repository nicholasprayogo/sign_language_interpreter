# Sign Language Interpreter
This project uses convolutional neural networks to train a model that could accurately interpret sign language alphabets.

## Setup
Install [tflearn](http://tflearn.org/installation/) and all other required packages.
For most situations, `pip install tflearn` or `pip3 install tflearn` works just fine.
Real-time video feed processing requires a decent graphics card.

## Training
To visualize data and train the model, run [train.py](https://github.com/nicholasprayogo/sign_language_interpreter/blob/master/train.py).

## Testing against real-time video feed
Run [test_videofeed.py](https://github.com/nicholasprayogo/sign_language_interpreter/blob/master/test_videofeed.py).

Note: cv2.VideoCapture(0) might need to be modified depending on your webcam device ID (usually 0 if only have 1 webcam).
