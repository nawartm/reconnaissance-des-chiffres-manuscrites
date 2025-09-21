# Handwritten Digit Recognition with Deep Learning

> *Objective: Teach a computer to recognize handwritten digits — just like a child learns to read numbers.*

This project contains **two Jupyter notebooks** implementing two types of neural networks to classify handwritten digits from the famous **MNIST** dataset:

- **`01-DNN-MNIST.ipynb`** → **Dense Neural Network (DNN)** — simple, effective, perfect for beginners.
- **`02-CNN-MNIST.ipynb`** → **Convolutional Neural Network (CNN)** — more powerful, designed specifically for image data.

---

## Why MNIST?

The **MNIST** dataset is the “Hello World” of Deep Learning. It contains:

- ✍️ **60,000 training images** (handwritten digits 0–9)
- ✍️ **10,000 test images**
- 📏 Black-and-white images, 28×28 pixels
- 🎯 Goal: Predict which digit is represented in each image

It’s the ideal training ground for understanding the basics of image classification with neural networks.

---

## Target Audience

| Audience | What They Will Find |
|----------|----------------------|
| **Students / AI Beginners** | A step-by-step tutorial with simple code, clear explanations, and visualizations to understand how neural networks work. |
| **Teachers / Trainers** | A complete pedagogical resource for teaching DNNs and CNNs, including evaluation, training history, confusion matrices, etc. |
| **Data Scientists / Developers** | A clean implementation using Keras/TensorFlow, easy to modify, extend, or compare — perfect for experimentation. |
| **Curious Non-Technical Readers** | Simple explanations, intuitive visuals, and a concrete demonstration of how machines “see” and “recognize” digits. |

---

## What You Will Learn

### ✅ Common to Both Notebooks:
- Load and normalize MNIST data
- Visualize training images
- Compile model with `Adam`, `sparse_categorical_crossentropy`, and `accuracy`
- Train model over 16 epochs
- Evaluate performance (accuracy, loss)
- Plot training history
- Display predictions (correct and incorrect)
- Generate a **confusion matrix**

### 🧱 01-DNN-MNIST.ipynb — Dense (Fully Connected) Network
- Simple architecture:
  - `Flatten()` → converts 28×28 image into a 784-value vector
  - Two hidden layers of 100 neurons with `ReLU` activation
  - Output layer of 10 neurons with `softmax` (probabilities for each digit)
- Expected accuracy: **~97.7%**

### 🧩 02-CNN-MNIST.ipynb — Convolutional Network
- Image-optimized architecture:
  - `Conv2D` layers → detect local patterns (edges, curves…)
  - `MaxPooling2D` layers → reduce spatial size while preserving important features
  - `Dropout` layers → prevent overfitting
  - `Flatten` + `Dense` → final classification
- Expected accuracy: **> 98.5%** (often ~99%)

---

## Technical Steps Implemented

### 1. Data Preparation
- Normalize pixel values to range [0, 1]
- For CNN: Add channel dimension (`reshape(-1, 28, 28, 1)`)

### 2. Model Construction
- Built using `keras.Sequential`
- Activation functions: `relu`, `softmax`
- Compiled with `sparse_categorical_crossentropy` (since labels are integers, not one-hot encoded)

### 3. Training
- `batch_size = 512`
- `epochs = 16`
- Real-time validation on test set during training

### 4. Evaluation & Visualization
- Compute final accuracy
- Plot training history (loss and accuracy on train/test)
- Display predictions (green = correct, red = incorrect)
- Generate normalized confusion matrix → identify where the model makes the most mistakes

---

## Technologies & Libraries Used

```python
import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

# Custom utility (adapt as needed)
import fidle.pwk as pwk  # → plot_images, plot_history, plot_confusion_matrix...
