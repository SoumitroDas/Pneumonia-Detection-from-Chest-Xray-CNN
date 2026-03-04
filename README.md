# Pneumonia Detection from Chest X-Ray using CNN

This repository contains a complete deep learning pipeline for detecting pneumonia from pediatric chest X-ray images using a Convolutional Neural Network (CNN) implemented in TensorFlow and Keras.

The project focuses on a clean and understandable end-to-end workflow including data loading, preprocessing, augmentation, model design, training, evaluation, and single-image inference.

---

## 📌 Problem Statement

Pneumonia is one of the leading causes of mortality in young children worldwide.  
Chest X-ray imaging is commonly used as an auxiliary tool for diagnosis.

This project aims to build a binary image classification model to automatically distinguish between:

- **NORMAL**
- **PNEUMONIA**

from chest X-ray images.

---

## 📂 Dataset

The dataset used in this project is from Kaggle:

**Title:** Labeled Chest X-Ray Images  
**Author:** Tolga Dinçer  
**Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/tolgadincer/labeled-chest-xray-images)  
**License:** CC0: Public Domain  
**Description:** Pediatric chest X-ray images labeled as NORMAL or PNEUMONIA for classification purposes.

**Suggested Citation (APA style):**

> Dinçer, T. (2020). *Labeled Chest X-Ray Images* [Dataset]. Kaggle. https://www.kaggle.com/datasets/tolgadincer/labeled-chest-xray-images

The dataset contains:

- 5,856 pediatric chest X-ray images
- Two main folders:
  - `train/`
  - `test/`
- Two classes:
  - `NORMAL`
  - `PNEUMONIA`

Images are collected from pediatric patients aged 1–5 years and are labeled at the patient level.

---

## ⚙️ Environment

The project is designed and tested using:

- Google Colab
- Python 3
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- scikit-learn

---

## 🧠 Model Architecture

A custom Convolutional Neural Network is implemented using the following block structure:

- Convolution → ReLU
- Max Pooling
- Convolution → ReLU
- Max Pooling
- Convolution → ReLU
- Max Pooling
- Flatten
- Fully connected layer
- Softmax output layer

The final layer uses softmax with two outputs for binary classification.

---

## 🏗️ Input Configuration

- Input image size: `224 × 224 × 3`
- Pixel normalization: `1/255`
- Batch size: `32`

---

## 🔄 Data Preprocessing and Augmentation

Training images are augmented using:

- Random zoom
- Small rotations
- Rescaling

No artificial augmentation is applied to the test set.

---

## 🚀 Training Strategy

- Training data is loaded using `flow_from_directory`
- Test data is kept separate
- Model is trained using:

```python
cnn.fit(training_set, validation_data=test_set, epochs=25)
