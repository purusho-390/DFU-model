# Detection of Diabetic Foot Ulcer using Deep Learning

## Overview

This project aims to develop a deep learning model for the detection of diabetic foot ulcers from medical images. Diabetic foot ulcers are a common complication of diabetes and can lead to serious infections and amputations if not detected and treated early. By leveraging deep learning techniques, we seek to provide a tool for early detection and intervention, ultimately improving patient outcomes.

## Dataset

We utilized a dataset consisting of medical images of diabetic foot ulcers, collected from various medical institutions. The dataset includes images of varying resolutions and qualities, annotated by medical professionals for ulcer presence and severity.

https://www.kaggle.com/datasets/purushomohan/dfu-wagners-classification

## Methodology

### Preprocessing
- Image resizing and normalization
- Data augmentation techniques to increase dataset size and model robustness

### Model Architecture
- Utilized a convolutional neural network (CNN) architecture for image classification
- Transfer learning using pre-trained models such as VGG, ResNet, or Inception for feature extraction
- Fine-tuning the pre-trained models on our dataset for optimal performance

### Training
- Split the dataset into training, validation, and test sets
- Utilized cross-validation techniques for model evaluation
- Employed appropriate loss functions and optimization algorithms
- Experimented with hyperparameter tuning to optimize model performance

## Results

Our model achieved promising results in the detection of diabetic foot ulcers, with an accuracy of [insert accuracy] on the test set. Further evaluation metrics such as precision, recall, and F1-score are provided in the accompanying documentation.

## Usage

### Requirements
- Python 3.x
- TensorFlow/Keras
- NumPy
- Pandas
- Matplotlib (for visualization)
- [Additional libraries]

### Instructions
1. Clone the repository:
# Detection of Diabetic Foot Ulcer using Deep Learning

## Overview

This project aims to develop a deep learning model for the detection of diabetic foot ulcers from medical images. Diabetic foot ulcers are a common complication of diabetes and can lead to serious infections and amputations if not detected and treated early. By leveraging deep learning techniques, we seek to provide a tool for early detection and intervention, ultimately improving patient outcomes.

## Dataset

We utilized a dataset consisting of medical images of diabetic foot ulcers, collected from various medical institutions. The dataset includes images of varying resolutions and qualities, annotated by medical professionals for ulcer presence and severity.

## Methodology

### Preprocessing
- Image resizing and normalization
- Data augmentation techniques to increase dataset size and model robustness

### Model Architecture
- Utilized a convolutional neural network (CNN) architecture for image classification
- Transfer learning using pre-trained models such as VGG, ResNet, or Inception for feature extraction
- Fine-tuning the pre-trained models on our dataset for optimal performance

### Training
- Split the dataset into training, validation, and test sets
- Utilized cross-validation techniques for model evaluation
- Employed appropriate loss functions and optimization algorithms
- Experimented with hyperparameter tuning to optimize model performance

## Results

Our model achieved promising results in the detection of diabetic foot ulcers, with an accuracy of 93% on the test set. Further evaluation metrics such as precision, recall, and F1-score are provided in the accompanying documentation.

## Usage

### Requirements
- Python 3.x
- TensorFlow/Keras
- NumPy
- Pandas
- Matplotlib (for visualization)

### Instructions
1. Clone the repository:
   https://github.com/purusho-390/dfu-model.git
2. Install the required dependencies:
   pip install -r requirements.txt
3. [Preprocess your data, if necessary]
4. Train the model:
   python dfu-training.py
5. Evaluate the model:
   python dfu-detection.py



