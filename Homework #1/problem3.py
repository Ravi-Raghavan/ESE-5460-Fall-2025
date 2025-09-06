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
def plot_images(X, y, start):
    plt.figure(figsize=(8, 8))
    for i in range(start, start + 16):
        offset = i - start
        plt.subplot(4, 4, offset+1)
        plt.imshow(X[i], cmap='gray')
        plt.title(f"Label: {y[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Plot first 16 images from the downsampled training set
# plot_images(X_train, y_train, 25000)
# plot_images(X_val, y_val)


## Part (b)
class embedding_t:
    def __init__(self):
        # initialize to appropriate sizes, fill with Gaussian entries
        mean = 0.0
        std = 0.01
        self.w = np.random.normal(loc = mean, scale = std, size = (4, 4, 8))
        self.b = np.random.normal(loc = mean, scale = std, size = (8,))

        # normalize to make the Frobenius norm of (w, b) equal to 1
        fro_norm = np.sqrt(np.sum(self.w ** 2) + np.sum(self.b ** 2))
        self.w = self.w / fro_norm
        self.b = self.b / fro_norm
    
    def zero_grad(self):
        # useful to delete the stored backprop gradients of the previous mini-batch before you start a new mini-batch
        self.dw, self.db = 0, 0

    # Shape of hl: B x 28 x 28
    def forward(self, hl):
        # Store Batch Value
        B = hl.shape[0]

        # We denote dimension of hl as B X 28 X 28 where B is the batch size
        # Step 1: Convert hl to B X 28 X 28 X 1
        hl = hl[:, :, :, None]

        # Step 2: Convert hl to B X 28 X 28 X 8
        hl = np.repeat(hl, repeats = 8, axis = -1)

        # We cache hl in forward because we need to compute it in backward
        self.hl = hl

        # Step 3: Form Sliding Windows: Shape is B x 25 x 25 x 1 x 4 x 4 x 8
        sliding_windows = np.lib.stride_tricks.sliding_window_view(hl, window_shape = (4, 4, 8), axis = (1, 2, 3))
        
        # Step 4: To capture Stride in our convolution, subset windows and we now have B x 7 x 7 x 4 x 4 x 8
        sliding_windows = sliding_windows[:, ::4, ::4, 0]

        # Step 5: Element-wise multiplication to get output of B x 7 x 7 x 8
        hl_plus_1 = (sliding_windows * self.w).sum(axis = (3, 4)) + self.b

        # Step 6: Convert to B x 392
        hl_plus_1 = hl_plus_1.reshape(B, -1)

        return hl_plus_1
    
    # dhl_plus_1 Shape: B x 392
    def backward(self, dhl_plus_1):
        # Store Batch Value
        B = dhl_plus_1.shape[0]

        # Step 1: Reshape as B x 7 x 7 x 8
        dhl_plus_1 = dhl_plus_1.reshape(B, 7, 7, 8)

        # Bring back self.hl and create sliding windows
        # Step 2: Form Sliding Windows: Shape is B x 25 x 25 x 1 x 4 x 4 x 8
        sliding_windows = np.lib.stride_tricks.sliding_window_view(self.hl, window_shape = (4, 4, 8), axis = (1, 2, 3))
        
        # Step 3: To capture Stride in our convolution, subset windows and we now have B x 7 x 7 x 4 x 4 x 8
        sliding_windows = sliding_windows[:, ::4, ::4, 0]

        # Step 4: Compute dw and db
        dhl_plus_1_reshaped = dhl_plus_1[:, :, :, None, None, :] # Reshape to B x 7 x 7 x 1 x 1 x 8
        dw = (sliding_windows * dhl_plus_1_reshaped).sum(axis = (0, 1, 2))
        db = dhl_plus_1_reshaped.sum(axis = (0, 1, 2, 3, 4))

        # Step 5: Store gradients for w and b
        self.dw, self.db = dw, db

        # Step 6: Compute dhl
        dhl = (self.w * dhl_plus_1_reshaped).sum(axis = -1) # B x 7 x 7 x 4 x 4 
        dhl = dhl.transpose(0, 1, 3, 2, 4).reshape(B, 28, 28)
        return dhl


l1 = embedding_t()

# Test Forward Pass
hl = np.random.randn(30000, 28, 28)
hl_plus_1 = l1.forward(hl)

# Test Backward Pass
dhl_plus_1 = np.random.randn(30000, 392)
l1.backward(dhl_plus_1)
