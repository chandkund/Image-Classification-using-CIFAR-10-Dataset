
# Image Classification using CIFAR-10 Dataset

This project aims to build an image classification model using the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. This project leverages deep learning techniques to classify these images into their respective categories.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Code Explanation](#code-explanation)
- [Model Evaluation](#model-evaluation)
- [License](#license)

## Project Overview

The CIFAR-10 dataset is commonly used for benchmarking image classification algorithms. This project includes the following steps:

1. Loading and preprocessing the CIFAR-10 dataset.
2. Building and training a convolutional neural network (CNN) model.
3. Evaluating the model's performance on test data.
4. Making predictions and visualizing results.

## Installation

To run this project, you will need Python along with the following libraries:

- `tensorflow`
- `keras`
- `numpy`
- `matplotlib`

You can install the required packages using `pip`:

```bash
pip install tensorflow keras numpy matplotlib
```

## Usage

1. Clone the repository:

    ```bash
    git clone https://github.com/chandkund/Image-Classification-using-CIFAR-10-Dataset.git
    cd Image-Classification-using-CIFAR-10-Dataset
    ```

2. Run the script to train and evaluate the model:

    ```bash
    python train_model.py
    ```

## Code Explanation

- **Import Libraries**:

    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    from tensorflow.keras.datasets import cifar10
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
    from tensorflow.keras.utils import to_categorical
    ```

- **Load and Preprocess Data**:

    ```python
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    # Convert class vectors to binary class matrices
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    ```

- **Build the CNN Model**:

    ```python
    model = models.Sequential([
    layers.Conv2D(32,(3,3),activation ='relu',input_shape = (32,32,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3),activation ='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3),activation ='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64,activation= 'relu'),
    layers.Dense(10,activation ='softmax')])
    ])
    ```

- **Compile and Train the Model**:

    ```python
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(train_images, train_labels, epochs=10,
                        validation_data=(test_images, test_labels))
    ```

- **Evaluate the Model**:

    ```python
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f'Test Accuracy: {test_acc}')
    ```

- **Visualize Training History**:

    ```python
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()
    ```

## Model Evaluation

After training the model, evaluate its performance on the test dataset. The evaluation metrics include accuracy and loss. Visualize the training and validation accuracy to understand the model's performance over epochs.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
