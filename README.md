# Dog vs Cat Classification using Transfer Learning

This repository contains a TensorFlow implementation for a deep learning model that classifies images into either dogs or cats using transfer learning with MobileNetV2.

## Introduction
This project utilizes the Kaggle dataset ["Cat & Dog Images for Classification"](https://www.kaggle.com/ashfakyeafi/cat-dog-images-for-classification), consisting of a large number of images of cats and dogs. Transfer learning is employed, leveraging a pre-trained MobileNetV2 model for feature extraction.

## Requirements
- TensorFlow
- TensorFlow Hub
- Matplotlib
- NumPy
- OpenCV
- Seaborn
- Kaggle API

## Setup and Dataset Download
To replicate the project, follow these steps:
1. Install the Kaggle API and download the dataset.
2. Import necessary libraries.
3. Preprocess the dataset, including resizing images and converting them into numpy arrays.

## Model Training
1. The MobileNetV2 model is imported from TensorFlow Hub as a feature extractor.
2. A new dense layer with softmax activation is added on top for classification.
3. The model is compiled with the Adam optimizer and sparse categorical crossentropy loss.
4. Model training is carried out on the preprocessed training data.

## Accuracy

The model achieved a test accuracy of 98.25% on the test set. This indicates its capability to effectively classify images of dogs and cats.

## Evaluation and Visualization
1. Model performance is evaluated on the test data, computing loss and accuracy.
2. Predictions are made on the test data.
3. Classification report and confusion matrix are generated for evaluation.
4. Training and validation loss/accuracy are visualized using matplotlib.

## Saving the Model
The trained model is saved using joblib for future use.
