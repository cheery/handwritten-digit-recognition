# Handwritten Digit Recognition

This project is a simple handwritten digit recognition application using a
Convolutional Neural Network (CNN) trained on the MNIST dataset. The user can
draw a digit on a window, and the application will predict the digit using the
trained model.

This project is a study project developed with the assistance of GPT. The
initial code provided by GPT required several modifications and tweaks to
improve the digit recognition accuracy.

Mainly, CNN network had to be extended from the initial,
because although it performed well on the training data, it didn't recognize mouse-drawn digits very well.
The digit recognizer had a bug where it transposed the input image, mismatching the training data orientation.
Additionally the input image had to be cropped and padded to match how the digits are given in the training data.

## Files

1. `cnn_model.py`: This file contains the CNN model architecture used for digit recognition.
2. `train.py`: This file is used to train the CNN model on the MNIST dataset and save the trained model.
3. `main.py`: This file is the main application that allows the user to draw digits and get predictions using the trained model.

## How to use

1. Install the required packages: `pygame`, `torch`, and `torchvision`.
2. Train the model by running `train.py`. This will train the CNN model on the MNIST dataset and save it as `trained_model.pt`.
3. Run `main.py` to start the application. Draw a digit on the window that appears, and the application will predict the digit using the trained model. The prediction will be printed in the console.

## Customization

You can modify the training parameters in `train.py` to change the number of
epochs, batch size, or other settings. Additionally, you can modify the CNN
model architecture in `cnn_model.py` to experiment with different
configurations.
