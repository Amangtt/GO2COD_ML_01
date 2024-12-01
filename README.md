# GO2COD_ML_01
Overview
This repository contains a Python script that implements a digit classification model using the MNIST dataset. The model is built with TensorFlow and Keras and evaluates its performance using accuracy, precision, and recall metrics.

Requirements
Make sure you have the following libraries installed:

TensorFlow
NumPy
Matplotlib
scikit-learn

Usage
1. Load the MNIST Dataset: The script begins by loading the MNIST dataset, which consists of 70,000 handwritten digits, divided into a training set and a test set.
2. Normalize the Data: The pixel values of the images are normalized to the range [0, 1] to improve training stability.
3. Define the Model: A simple neural network is defined with:
   A flattening layer to convert the 2D images into 1D arrays.
   A dense hidden layer with 128 neurons and ReLU activation.
   An output layer with 10 neurons (one for each digit) and a linear activation function.
4. Compile the Model: The model is compiled with Sparse Categorical Crossentropy loss, Adam optimizer, and accuracy as a metric.
5. Train the Model: The model is trained on the training set for 10 epochs.
6. Make Predictions: Predictions are made on the test set.
7. Evaluate the Model: The model's performance is evaluated using accuracy, precision, and recall metrics.
8. Display Predictions: The script displays some of the test images along with their predicted and true labels.
   
Results
After running the script, you will see the following outputs:
        Test accuracy,
        Precision,
        Recall,
        A grid of images showing predicted versus true labels.
