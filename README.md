# Context Recognition

A deep learning project for environmental sound classification using convolutional neural networks.

## Overview

This project implements a sound classification system capable of recognizing different environmental contexts based on audio data. It uses the ESC-50 dataset (Environmental Sound Classification) and implements convolutional neural networks with PyLearn2 and Theano.

## Dataset

The project uses the [ESC-50 dataset](https://github.com/karolpiczak/ESC-50), a labeled collection of 2000 environmental audio recordings suitable for benchmarking methods of environmental sound classification.

## Project Structure

- `Dataset.ipynb`: Data processing utilities and dataset preparation
- `Net-DoubleConv.ipynb`: Implementation of the convolutional neural network architecture
- `EDA on ESC50 Dataset.ipynb`: Exploratory data analysis on the ESC-50 dataset
- `Evaluation.ipynb`: Model evaluation utilities and metrics

## Requirements

The project requires:
- Python 2.7
- PyLearn2
- Theano
- NumPy
- Pandas
- scikit-learn
- IPython/Jupyter

## How to Use

1. Clone the repository
2. Download the ESC-50 dataset
3. Run the notebooks in the following order:
   - First, explore the dataset using `EDA on ESC50 Dataset.ipynb`
   - Prepare the dataset using `Dataset.ipynb`
   - Train the model with `Net-DoubleConv.ipynb`
   - Evaluate the model using `Evaluation.ipynb`

## Model Architecture

The model uses a double convolutional neural network architecture with:
- Multiple convolutional layers with ReLU activation
- Max pooling layers
- Dropout for regularization
- Softmax output layer for classification

## Results

The model is evaluated on sound classification accuracy across different environmental contexts.

