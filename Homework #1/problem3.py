# Ravi Raghavan, Homework #1, Problem 3

## Part (a)
import torchvision as thv
train = thv.datasets.MNIST("./", download=True, train=True)
val = thv.datasets.MNIST("./", download=True, train=False)
print(train.data.shape, len(train.targets))
print(val.data.shape, len(val.targets))

### Convert everything to numpy. From here on out, no Torch/Other Deep Learning Library 
import numpy as np
X_train = train.data.numpy()
y_train = train.targets.numpy()
X_val = val.data.numpy()
y_val = val.targets.numpy()

print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)

### Take 50% of each class from training and validation
def downsample(X, y):
    # Store downsampled X and y
    X_downsampled, y_downsampled = [], []

    # Iterate through the unique labels of y
    for label in np.unique(y):
        indices = np.where(y == label)[0] # Fetch indices where y == label
        indices = np.sort(indices)
        half = len(indices) // 2 # Get half
        X_downsampled.append(X[indices[:half]])
        y_downsampled.append(y[indices[:half]])
    
    return np.concatenate(X_downsampled, axis = 0), np.concatenate(y_downsampled, axis = 0)

X_train, y_train = downsample(X_train, y_train)
X_val, y_val = downsample(X_val, y_val)

print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)

## Plot the images of a few images in the dataset just to see if label is right
# Function to plot a grid of images with labels
import matplotlib.pyplot as plt
def plot_images(X, y, num_images=16):
    plt.figure(figsize=(8, 8))
    for i in range(num_images):
        plt.subplot(4, 4, i+1)
        plt.imshow(X[i], cmap='gray')
        plt.title(f"Label: {y[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Plot first 16 images from the downsampled training set
# plot_images(X_train, y_train)
# plot_images(X_val, y_val)
