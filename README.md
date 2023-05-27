# Super-Resolution Image Generation with Deep Learning
This repository contains code for generating high-resolution RGB images from low-resolution hyperspectral images using deep learning techniques. The model is trained using a dataset of hyperspectral images and their corresponding high-resolution RGB images.

# Dataset
The dataset used for training and testing the model consists of hyperspectral images stored in the complete_ms_data folder. Each hyperspectral image is divided into patches of size 64x64 pixels with a stride of 32 pixels. The dataset is split into a training set (80% of the data) and a testing set (20% of the data).

# Data Preprocessing
To simulate low-resolution images, an 8x8 averaging filter is applied to each band of the ground truth hyperspectral images. This generates low-resolution hyperspectral images of size 8x8x31. Additionally, high-resolution RGB images are obtained by averaging the spectral bands in groups of 10, resulting in images of size 64x64x3.

# Model Architecture
The model architecture used for super-resolution image generation is a convolutional neural network. It takes two inputs: the high-resolution RGB image and the low-resolution hyperspectral image. The low-resolution image is upsampled using a transpose convolution layer to match the size of the high-resolution input. Both inputs are then concatenated and passed through a series of convolutional and transpose convolutional layers. The final output is a high-resolution RGB image.

# Training
The model is trained using the Mean Squared Error (MSE) loss function and the Adam optimizer. The training process involves optimizing the model's weights and biases to minimize the difference between the generated high-resolution RGB images and the ground truth RGB images. The model is trained for 10 epochs with a batch size of 32.

# Usage
To run the code and train the super-resolution model, follow these steps:

* Ensure that all the required libraries are installed, including NumPy, Matplotlib, scikit-learn, OpenCV, and Keras.
* Set the path to the dataset in the dataset_path variable.
* Run the code and wait for the training process to complete.
* Once the model is trained, you can use it to generate high-resolution RGB images from low-resolution hyperspectral images.
* * *
***Note: The code provided assumes that you have the necessary dataset and that it is structured as described above. Adjustments may be needed based on your specific dataset and requirements.***
