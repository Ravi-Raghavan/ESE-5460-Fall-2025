# Ravi Raghavan, Homework #1, Problem 3

# Throughout the entire problem, I set the random state here for reproducability
random_state = 42

## Part (a)

### Load Data from torchvision using the Code provided by the assignment
import torchvision as thv
train = thv.datasets.MNIST("./", download=True, train=True)
val = thv.datasets.MNIST("./", download=True, train=False)
print("Part (a): Print out shape of train.data, train.targets, val.data, and val.targets")
print(train.data.shape, len(train.targets), val.data.shape, len(val.targets))

### Convert the above PyTorch Tensors to numpy. From here on out, NO Torch/Other Deep Learning Library 
### PyTorch will ONLY be used at the VERY END to verify my results
import numpy as np
np.random.seed(random_state) # set random state, throughout the entire problem, for reproducability
X_train = train.data.numpy()
y_train = train.targets.numpy()
X_val = val.data.numpy()
y_val = val.targets.numpy()
print("Part (a): Print out shape of X_train, y_train, X_val, and y_val after converting to numpy")
print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)

### Normalize Data
X_train = X_train.astype(np.float64) / 255.0
X_val = X_val.astype(np.float64) / 255.0

### Given X and y, where X contains training samples and y contains labels, keep only 50% of each class! 
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

### Perform downsampling on X_train, y_train, X_val, y_val
X_train, y_train = downsample(X_train, y_train)
X_val, y_val = downsample(X_val, y_val)
print("Part (a): Print out shape of X_train, y_train, X_val, and y_val after downsampling")
print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)

## Plot the images of a few images in the dataset just to see if label is right
# Function to plot a grid of images with labels
import matplotlib.pyplot as plt
def plot_images_random(X, y, file_title, fig_title=None, n_images=16):
    indices = np.random.choice(len(X), size=n_images, replace=False)
    plt.figure(figsize=(8, 8))

    if fig_title:
        plt.suptitle(fig_title, fontsize=16)

    for i, idx in enumerate(indices):
        plt.subplot(4, 4, i+1)
        plt.imshow(X[idx], cmap='gray')
        plt.title(f"Label: {y[idx]}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(file_title, dpi=300)

# Plot first 16 images from the downsampled training set
plot_images_random(X_train, y_train, "3a_train_images.png", "Training Images w/ Labels")
plot_images_random(X_val, y_val, "3a_val_images.png", "Validation Images w/ Labels")

## Problem 3, Part (b)
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
        dhl = dhl.transpose(0, 1, 3, 2, 4).reshape(B, 28, 28) # B x 28 x 28
        return dhl

## Problem 3, Part (c)
class linear_t:
    def __init__(self):
        # initialize to appropriate sizes, fill with Gaussian entries
        mean = 0.0
        std = 0.01
        self.w = np.random.normal(loc = mean, scale = std, size = (10, 392))
        self.b = np.random.normal(loc = mean, scale = std, size = (10,))

        # normalize to make the Frobenius norm of (w, b) equal to 1
        fro_norm = np.sqrt(np.sum(self.w ** 2) + np.sum(self.b ** 2))
        self.w = self.w / fro_norm
        self.b = self.b / fro_norm
    
    def zero_grad(self):
        # useful to delete the stored backprop gradients of the previous mini-batch before you start a new mini-batch
        self.dw, self.db = 0, 0
    
    def forward(self, hl):
        # Cache hl in forward because needed for back
        self.hl = hl

        # Compute h_{l + 1}
        hl_plus_1 = hl @ self.w.T + self.b

        return hl_plus_1
    
    # Shape of dhl_plus_1: B x 10
    def backward(self, dhl_plus_1):
        # Compute dhl
        dhl = dhl_plus_1 @ self.w
        
        # Compute db
        db = dhl_plus_1.sum(axis = 0)
        self.db = db

        # Compute dw
        dw = dhl_plus_1.T @ self.hl
        self.dw = dw

        return dhl

## Problem 3, part (d)
class relu_t:
    def __init__(self):
        pass
    def zero_grad(self):
        pass
    def forward(self, hl):
        # Cache hl in forward because needed for back
        self.hl = hl

        # Compute h_{l + 1}
        hl_plus_1 = np.maximum(0, hl)

        return hl_plus_1
    
    def backward(self, dhl_plus_1):
        return np.where(self.hl < 0, 0, dhl_plus_1)

## Problem 3, part (e)
class softmax_cross_entropy_t:
    def __init__(self):
        pass

    def zero_grad(self):
        pass

    def forward(self, hl, y):
        # Cache hl in forward because needed for back
        self.hl = hl

        # Flatten y as sanity check 
        y = y.flatten()
        self.y = y

        # Step 1: Compute hl_plus_1 after Softmax
        exp_hl = np.exp(hl)
        hl_plus_1 = exp_hl / np.sum(exp_hl, axis = 1, keepdims=True)

        self.hl_plus_1 = hl_plus_1
        
        # Step 2: Compute average loss over minibatch 
        B = hl.shape[0]

        # Pick the probabilities corresponding to correct labels
        correct_probs = hl_plus_1[np.arange(B), y]
        ell = -np.mean(np.log(correct_probs + 1e-12)) #Adding 1e-12 for numerical stability

        # Step 3: Compute classification error 
        y_pred = np.argmax(hl_plus_1, axis=1)
        error = np.mean(y_pred != y)

        return ell, error

    def backward(self):
        B = self.hl.shape[0]

        # Get output from softmax
        softmax_output = self.hl_plus_1

        # Create one hot labels
        y_one_hot = np.zeros_like(softmax_output)
        y_one_hot[np.arange(B), self.y] = 1

        dhl = (softmax_output - y_one_hot) / B
        return dhl

## Problem 3, Part (f): Check implementation of forward and backward functionalities for all layers

### Checking Linear Layer Backward
#indices_W must be passed as a list of tuples
#indices_b must be passed as list
#indices_h must be passed as list
def check_backward_linear(k, indices_W, indices_b, indices_h):
    layer = linear_t() # Linear Layer
    hl = np.random.randn(1, 392) # Shape: 1 x 392

    # Compute forward pass of linear layer
    layer.forward(hl)

    # Get Weight and Bias of linear layer
    W = layer.w # Shape: 10 x 392
    b = layer.b # Shape: (10,)

    # Set up dhl_plus_1
    dhl_plus_1 = np.zeros(shape = (1, 10))
    dhl_plus_1[0, k] = 1 # Shape: 1 x 10

    # Compute backward
    dhl = layer.backward(dhl_plus_1) # Shape: 1 x 392
    dw = layer.dw # Shape: 10 x 392
    db = layer.db # Shape: (10,)

    # Verify dw
    for i, j in indices_W:
        eps = np.zeros(shape = W.shape)
        eps[i, j] = np.random.normal(loc = 0.0, scale = 1e-8)
        deriv_W = ((hl @ (W + eps).T + b) - (hl @ (W - eps).T + b))[0, k] / (2 * eps)[i, j]
        np.testing.assert_allclose(dw[i, j], deriv_W, rtol=1e-6, atol=1e-6)
    
    for i in indices_b:
        eps = np.zeros(shape = b.shape)
        eps[i] = np.random.normal(loc = 0.0, scale = 1e-8)
        deriv_b = ((hl @ W.T + (b + eps)) - (hl @ W.T + (b - eps)))[0, k] / (2 * eps)[i]
        np.testing.assert_allclose(db[i], deriv_b, rtol=1e-6, atol=1e-6)
    
    for i in indices_h:
        eps = np.zeros(shape = hl.shape)
        eps[0, i] = np.random.normal(loc = 0.0, scale = 1e-8)
        deriv_h = (((hl + eps) @ W.T + b) - ((hl - eps) @ W.T + b))[0, k] / (2 * eps)[0, i]
        np.testing.assert_allclose(dhl[0, i], deriv_h, rtol=1e-6, atol=1e-6)

def test_backward_linear_random_indices():
    rng = np.random.default_rng()

    # Sample 5 values of k
    k_values = rng.choice(10, size=5, replace=False)

    for k in k_values:
        # For W (10 x 392), pick 10 random (i, j) pairs
        indices_W = [(rng.integers(0, 10), rng.integers(0, 392)) for _ in range(10)]
        
        # For b (10,), pick 3 random indices
        indices_b = rng.choice(10, size=3, replace=False).tolist()
        
        # For h (392,), pick 4 random indices
        indices_h = rng.choice(392, size=4, replace=False).tolist()

        # Run the gradient check for this k
        check_backward_linear(k, indices_W, indices_b, indices_h)

test_backward_linear_random_indices()

### Checking ReLU Backward Propagation
def check_backward_relu(k, indices_h):
    layer = relu_t()
    hl = np.random.randn(1, 10)

    # Compute forward pass
    layer.forward(hl)

    # Set up dhl_plus_1
    dhl_plus_1 = np.zeros(shape = (1, 10))
    dhl_plus_1[0, k] = 1 # Shape: 1 x 10

    # Compute backward
    dhl = layer.backward(dhl_plus_1)

    for i in indices_h:
        eps = np.zeros(shape = hl.shape)
        eps[0, i] = np.random.normal(loc = 0.0, scale = 1e-8)
        deriv_h = ((np.maximum(0, hl + eps)) - (np.maximum(0, hl - eps)))[0, k] / (2 * eps)[0, i]
        np.testing.assert_allclose(dhl[0, i], deriv_h, rtol=1e-6, atol=1e-6)

def test_backward_relu_random_indices():
    rng = np.random.default_rng()

    # Sample 5 values of k
    k_values = rng.choice(10, size=5, replace=False)

    for k in k_values:
        # For h (10,), pick 4 random indices
        indices_h = rng.choice(10, size=4, replace=False).tolist()

        # Run the gradient check for this k
        check_backward_relu(k, indices_h)

test_backward_relu_random_indices()

### Test softmax_cross_entropy_t Function
def softmax_cross_entropy_utility(hl, y):
    # Flatten y as sanity check 
    y = y.flatten()

    # Step 1: Compute hl_plus_1 after Softmax
    exp_hl = np.exp(hl)
    hl_plus_1 = exp_hl / np.sum(exp_hl, axis = 1, keepdims=True)
    
    # Step 2: Compute average loss over minibatch 
    B = hl.shape[0]

    # Pick the probabilities corresponding to correct labels
    correct_probs = hl_plus_1[np.arange(B), y]
    ell = -np.mean(np.log(correct_probs + 1e-12)) #Adding 1e-12 for numerical stability

    y_pred = np.argmax(hl_plus_1, axis=1)
    error = np.mean(y_pred != y)

    return ell, error

def check_backward_softmax(indices_h):
    layer = softmax_cross_entropy_t()
    hl = np.random.randn(1, 10)
    y = np.random.randint(0, 10, size=(1,))

    # Compute forward pass
    layer.forward(hl, y)

    # Compute backward
    dhl = layer.backward()

    for i in indices_h:
        eps = np.zeros(shape = hl.shape)
        eps[0, i] = np.random.normal(loc = 0.0, scale = 1e-8)
        LHS, _ = softmax_cross_entropy_utility(hl + eps, y)
        RHS, _ = softmax_cross_entropy_utility(hl - eps, y)
        diff = LHS - RHS
        deriv_h = diff / (2 * eps)[0, i]
        np.testing.assert_allclose(dhl[0, i], deriv_h, rtol=1e-6, atol=1e-6)

def test_backward_softmax_random_indices():
    rng = np.random.default_rng()

    # For h (10,), pick 4 random indices
    indices_h = rng.choice(10, size=5, replace=False).tolist()

    # Run the gradient check for this k
    check_backward_softmax(indices_h)
    
test_backward_softmax_random_indices()

### Test embedding_t Function
def embedding_utility(hl, W, b):
    # Store Batch Value
    B = hl.shape[0]

    # We denote dimension of hl as B X 28 X 28 where B is the batch size
    # Step 1: Convert hl to B X 28 X 28 X 1
    hl = hl[:, :, :, None]

    # Step 2: Convert hl to B X 28 X 28 X 8
    hl = np.repeat(hl, repeats = 8, axis = -1)

    # Step 3: Form Sliding Windows: Shape is B x 25 x 25 x 1 x 4 x 4 x 8
    sliding_windows = np.lib.stride_tricks.sliding_window_view(hl, window_shape = (4, 4, 8), axis = (1, 2, 3))
    
    # Step 4: To capture Stride in our convolution, subset windows and we now have B x 7 x 7 x 4 x 4 x 8
    sliding_windows = sliding_windows[:, ::4, ::4, 0]

    # Step 5: Element-wise multiplication to get output of B x 7 x 7 x 8
    hl_plus_1 = (sliding_windows * W).sum(axis = (3, 4)) + b

    # Step 6: Convert to B x 392
    hl_plus_1 = hl_plus_1.reshape(B, -1)

    return hl_plus_1

def check_backward_embedding(t, indices_W, indices_b, indices_h):
    layer = embedding_t()
    hl = np.random.randn(1, 28, 28)

    # Compute forward pass
    layer.forward(hl)

    # Set up backward pass
    W = layer.w # Shape: 4 x 4 x 8
    b = layer.b # Shape: (8,)

    # Set up dhl_plus_1
    dhl_plus_1 = np.zeros(shape = (1, 392))
    dhl_plus_1[0, t] = 1 # Shape: 1 x 392

    # Compute backward
    dhl = layer.backward(dhl_plus_1)
    dw = layer.dw # Shape: 4 x 4 x 8
    db = layer.db # Shape: (8,)

    # Verify dw
    for i, j, k in indices_W:
        eps = np.zeros(shape = W.shape)
        eps[i, j, k] = np.random.normal(loc = 0.0, scale = 1e-8)
        deriv_W = (embedding_utility(hl, W + eps, b) - embedding_utility(hl, W - eps, b))[0, t] / (2 * eps)[i, j, k]
        np.testing.assert_allclose(dw[i, j, k], deriv_W, rtol=1e-6, atol=1e-6)
    
    for i in indices_b:
        eps = np.zeros(shape = b.shape)
        eps[i] = np.random.normal(loc = 0.0, scale = 1e-8)
        deriv_b = (embedding_utility(hl, W, b + eps) - embedding_utility(hl, W, b - eps))[0, t] / (2 * eps)[i]
        np.testing.assert_allclose(db[i], deriv_b, rtol=1e-6, atol=1e-6)
    
    for i, j in indices_h:
        eps = np.zeros(shape = hl.shape)
        eps[0, i, j] = np.random.normal(loc = 0.0, scale = 1e-8)
        deriv_h = (embedding_utility(hl + eps, W, b) - embedding_utility(hl - eps, W, b))[0, t] / (2 * eps)[0, i, j]
        np.testing.assert_allclose(dhl[0, i, j], deriv_h, rtol=1e-6, atol=1e-6)

def test_backward_embedding_random_indices():
    rng = np.random.default_rng() # Set seed for reproducability

    # Sample 5 values of k
    t_values = rng.choice(10, size=5, replace=False)

    for t in t_values:
        # For W (4 x 4 x 8), pick 10 random (i, j) pairs
        indices_W = [(rng.integers(0, 4), rng.integers(0, 4), rng.integers(0, 8)) for _ in range(10)]
        
        # For b (8,), pick 3 random indices
        indices_b = rng.choice(8, size=3, replace=False).tolist()
        
        # For h (28, 28), pick 4 random indices
        indices_h = [(rng.integers(0, 28), rng.integers(0, 28)) for _ in range(4)]

        # Run the gradient check for this k
        check_backward_embedding(t, indices_W, indices_b, indices_h)

test_backward_embedding_random_indices()

## part (g) and (h)
### Initialize all Layers
l1, l2, l3, l4 = embedding_t(), linear_t(), relu_t(), softmax_cross_entropy_t()
l1_weights_init, l1_bias_init = l1.w, l1.b
l2_weights_init, l2_bias_init = l2.w, l2.b

batch_indices = np.empty((10000, 32), dtype=np.int64)
for i in range(10000):
    batch_indices[i] = np.random.choice(X_train.shape[0], size=32, replace=False)



### Part (i): Test Implementation using PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Define class for Embedding Layer
class EmbeddingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels = 1,
            out_channels = 8,
            kernel_size = 4,
            stride = 4,
            bias = True
        )

        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            l1_weights_init_reshaped = l1_weights_init.transpose(2, 0, 1)[:, None, :, :]
            self.conv.weight.copy_(torch.from_numpy(l1_weights_init_reshaped))
            self.conv.bias.copy_(torch.from_numpy(l1_bias_init))
    
    # x: B X 28 X 28
    def forward(self, x):
        x = x.unsqueeze(1) # Becomes B X 1 X 28 X 28
        x = self.conv(x) # Becomes B X 8 x 7 x 7
        x = x.permute(0, 2, 3, 1) # Becomes B x 7 x 7 x 8
        x = x.flatten(start_dim = 1) # Becomes B X 392
        return x

# Define Overall Neural Network
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.embed = EmbeddingLayer()
        self.fc = nn.Linear(392, 10)
    
    def _init_fc(self):
        with torch.no_grad():
            self.conv.weight.copy_(torch.from_numpy(l2_weights_init))
            self.conv.bias.copy_(torch.from_numpy(l2_bias_init))
    
    def forward(self, x):
        x = self.embed(x)
        x = self.fc(x)
        x = F.relu(x)

        return x

## Hyperparameters
batch_size = 32
lr = 0.1
epochs = 10000

# Convert NumPy arrays to torch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

# Set up model, loss function, and optimizer
model = NN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr)

def validate_torch(X, y):
    model.eval()  # set model to evaluation mode
    tot_loss = 0.0
    tot_error = 0.0

    with torch.no_grad():  # disable gradient computation
        for i in range(0, X.shape[0], batch_size):
            # Get batch
            X_batch = X[i: i + batch_size]
            y_batch = y[i: i + batch_size]

            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # Compute batch error (percentage misclassified)
            _, predicted = torch.max(outputs, 1)
            error = (predicted != y_batch).float().mean().item()

            # Accumulate weighted loss and error
            tot_loss += loss.item() * X_batch.size(0)
            tot_error += error * X_batch.size(0)
    
    # Average over all validation samples
    avg_loss = tot_loss / X.shape[0]
    avg_error = tot_error / X.shape[0]

    return avg_loss, avg_error

# Begin the training + validation loop
training_losses = []
training_errors = []
validation_losses = []
validation_errors = []
for epoch in range(epochs):
    # Run Training
    model.train()
    
    # Randomly sample a batch
    indices = batch_indices[epoch, :].flatten()
    batch_X = X_train_tensor[indices]
    batch_y = y_train_tensor[indices]
    
    # Zero out gradients
    optimizer.zero_grad()

    ## Obtain loss and do back propagation
    outputs = model(batch_X)
    loss = criterion(outputs, batch_y)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        avg_train_loss, avg_train_error = validate_torch(X_train_tensor, y_train_tensor)
        print(f"Epoch: {epoch + 1}, Training Loss: {avg_train_loss}, Training Error: {avg_train_error}")
        training_losses.append(avg_train_loss)
        training_errors.append(avg_train_error)

    if (epoch + 1) % 100 == 0:
        avg_val_loss, avg_val_error = validate_torch(X_val_tensor, y_val_tensor)
        print(f"Epoch: {epoch + 1}, Validation Loss: {avg_val_loss}, Validation Error: {avg_val_error}")
        validation_losses.append(avg_val_loss)
        validation_errors.append(avg_val_error)        

# Plot training + validation losses
plt.figure(figsize=(8, 5))
plt.plot(training_losses, label='Training Loss[PyTorch]')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss vs Iteration')
plt.legend()
plt.grid(True)
plt.savefig('training_loss_plot_torch.png', dpi=300)

plt.figure(figsize=(8, 5))
plt.plot(validation_losses, label='Validation Loss[PyTorch]')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Validation Loss vs Iteration')
plt.legend()
plt.grid(True)
plt.savefig('validation_loss_plot_torch.png', dpi=300)

# Plot training + validation errors
plt.figure(figsize=(8, 5))
plt.plot(training_errors, label='Training Error[PyTorch]')
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title('Training Error vs Iteration')
plt.legend()
plt.grid(True)
plt.savefig('training_error_plot_torch.png', dpi=300)

plt.figure(figsize=(8, 5))
plt.plot(validation_errors, label='Validation Error[PyTorch]')
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title('Validation Error vs Iteration')
plt.legend()
plt.grid(True)
plt.savefig('validation_error_plot_torch.png', dpi=300)

# Print Final Metrics
import pandas as pd
final_metrics = {
    "Training Error": [training_errors[-1]],
    "Training Loss": [training_losses[-1]],
    "Validation Error": [validation_errors[-1]],
    "Validation Loss": [validation_losses[-1]]
}

# Convert to DataFrame
df = pd.DataFrame(final_metrics)

# Save to CSV
df.to_csv("final_metrics_torch.csv", index=False)