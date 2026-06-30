"""
MNIST Image Processing and Visualization Tool

This script provides functionality for processing and visualizing MNIST handwritten digit images.
The main purpose is to help visualize and analyze MNIST dataset images during machine learning
tasks such as classification or clustering.
"""
import gzip
import math
import pickle
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

def get_MNIST_data():
    """
    Loads and preprocesses the MNIST dataset from a gzipped pickle file.

    The MNIST dataset is a collection of handwritten digits widely used for training
    various image processing systems. It contains 70,000 images in total.

    Process:
    1. Opens and reads a compressed pickle file containing MNIST data
    2. Combines training and validation sets into a single training set
    3. Returns separate arrays for training and test data

    Returns:
        tuple: Contains four numpy arrays:
            - train_x (numpy.ndarray): Training images, shape (60000, 784)
                        Each row represents a 28x28 pixel image as a vector of 784 pixels.
            - train_y (numpy.ndarray): Training labels, shape (60000,)
                        Contains digit labels (0-9)
            - test_x (numpy.ndarray): Test images, shape (10000, 784)
                        Same format as train_x
            - test_y (numpy.ndarray): Test labels, shape (10000, )
                        Same format as train_y

    Dependencies:
        - gzip: For reading compressed files
        - pickle: For loading Python objects
        - numpy: For array operations
    """

    with gzip.open('Datasets/mnist.pkl.gz', 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    f.close()
    train_set, valid_set, test_set = data
    train_x, train_y = train_set
    valid_x, valid_y = valid_set
    train_x = np.vstack((train_x, valid_x))
    train_y = np.append(train_y, valid_y)
    test_x, test_y = test_set
    return train_x, train_y, test_x, test_y

def plot_images(X):
    """
    Displays multiple MNIST digit images in a grid layout.

    This function takes a set of MNIST images stored as flattened arrays and displays
    them in a nearly-square grid arrangement. Each image is reshaped from a 1D array
    of 784 pixels into its original 28x28 pixel format and displayed in grayscale.

    Args:
        X (numpy.ndarray): Array of image data. Can be either:
            - 1D array of 784 pixels (single image)
            - 2D array of shape (n, 784) where n is the number of images
    """

    if X.ndim == 1:
        X = np.array([X])
    num_images = X.shape[0]
    num_rows = math.floor(math.sqrt(num_images))
    num_cols = math.ceil(num_images/num_rows)
    for i in range(num_images):
        reshaped_image = X[i,:].reshape(28,28)
        plt.subplot(num_rows, num_cols, i+1)
        plt.imshow(reshaped_image, cmap = cm.Greys_r)
        plt.axis('off')
    plt.show()

if __name__ == '__main__':
    # Load MNIST data:
    train_x, train_y, test_x, test_y = get_MNIST_data()
    print(train_x.shape)
    # Plot the first 20 images of the training set.
    print(test_x.shape)
    plot_images(train_x[0, :])
    plot_images(train_x[0:20, :])