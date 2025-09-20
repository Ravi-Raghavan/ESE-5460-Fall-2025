# Ravi Raghavan, Homework #1, Problem 3

# Throughout the entire problem, I set the random state here for reproducability
random_state = 42

## Part (a)
### Load Data from torchvision using the Code provided by the assignment.
import torchvision as thv
train = thv.datasets.MNIST("./", download=True, train=True)
val = thv.datasets.MNIST("./", download=True, train=False)
print("Part (a): Print out shape of train.data, train.targets, val.data, and val.targets")
print(train.data.shape, len(train.targets), val.data.shape, len(val.targets))

### Convert the above PyTorch Tensors to numpy. From here on out, NO Torch/Other Deep Learning Library 
### PyTorch will ONLY be used at the VERY END to verify my results
import numpy as np
np.random.seed(random_state) # set random state, throughout the entire problem, for reproducability
X_train, y_train, X_val, y_val = train.data.numpy(), train.targets.numpy(), val.data.numpy(), val.targets.numpy()
print("Part (a): Print out shape of X_train, y_train, X_val, and y_val after converting to numpy")
print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)

### Convert X_train and X_val to np.float64
X_train, X_val = X_train.astype(np.float64), X_val.astype(np.float64)

### Given X and y, where X contains training samples and y contains labels, keep only 50% of each class! 
def downsample(X, y):
    # Store downsampled X and y
    X_downsampled, y_downsampled = [], []
    for label in np.unique(y):
        indices = np.where(y == label)[0] # Fetch indices where y == label
        indices = np.sort(indices)
        half = len(indices) // 2
        X_downsampled.append(X[indices[:half]])
        y_downsampled.append(y[indices[:half]])
    
    return np.concatenate(X_downsampled, axis = 0), np.concatenate(y_downsampled, axis = 0)

### Perform downsampling on X_train, y_train, X_val, y_val
X_train, y_train = downsample(X_train, y_train)
X_val, y_val = downsample(X_val, y_val)
print("Part (a): Print out shape of X_train, y_train, X_val, and y_val after downsampling")
print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)

### Pre-processing: Step 1: Divide all pixels by 255 to get images in [0, 1].
X_train, X_val = X_train / 255.0,  X_val / 255.0

## Plot the images of a few images in the dataset just to see if label is right
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

# Plot first 16 images from the downsampled training and validation sets
plot_images_random(X_train, y_train, "3a_train_images.png", "Training Images w/ Labels")
plot_images_random(X_val, y_val, "3a_val_images.png", "Validation Images w/ Labels")

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

## Problem 3, Part (b): Embedding Layer
class embedding_t:
    def __init__(self):
        # initialize to appropriate sizes, fill with Gaussian entries
        mean, std = 0.0, 0.01
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

        # We denote dimension of hl as B X 28 X 28 where B is the batch size. We will also be caching it for use in backward propagatino
        # Step 1: Convert hl to B X 28 X 28 X 1
        self.hl = hl[:, :, :, None]

        # Step 2: Convert hl to B X 28 X 28 X 8
        self.hl = np.repeat(self.hl, repeats = 8, axis = -1)

        # Step 3: Form Sliding Windows: Shape is B x 25 x 25 x 1 x 4 x 4 x 8. Cache sliding_windows for use in backward
        self.sliding_windows = np.lib.stride_tricks.sliding_window_view(self.hl, window_shape = (4, 4, 8), axis = (1, 2, 3))
        
        # Step 4: To capture Stride in our convolution, subset windows and we now have B x 7 x 7 x 4 x 4 x 8
        self.sliding_windows = self.sliding_windows[:, ::4, ::4, 0]

        # Step 5: Element-wise multiplication to get output of B x 7 x 7 x 8
        hl_plus_1 = (self.sliding_windows * self.w).sum(axis = (3, 4)) + self.b

        # Step 6: Convert to B x 392
        hl_plus_1 = hl_plus_1.reshape(B, -1)

        return hl_plus_1
    
    # dhl_plus_1 Shape: B x 392
    def backward(self, dhl_plus_1):
        # Store Batch Value
        B = dhl_plus_1.shape[0]

        # Step 1: Reshape as B x 7 x 7 x 8
        dhl_plus_1 = dhl_plus_1.reshape(B, 7, 7, 8)

        # Step 2: Compute dw and db
        dhl_plus_1_reshaped = dhl_plus_1[:, :, :, None, None, :] # Reshape to B x 7 x 7 x 1 x 1 x 8
        dw = (self.sliding_windows * dhl_plus_1_reshaped).sum(axis = (0, 1, 2))
        db = dhl_plus_1_reshaped.sum(axis = (0, 1, 2, 3, 4))

        # Step 3: Store gradients for w and b
        self.dw, self.db = dw, db

        # Step 4: Compute dhl
        dhl = (self.w * dhl_plus_1_reshaped).sum(axis = -1) # B x 7 x 7 x 4 x 4 
        dhl = dhl.transpose(0, 1, 3, 2, 4).reshape(B, 28, 28) # B x 28 x 28
        return dhl

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

## Problem 3, Part (c): Linear Layer
class linear_t:
    def __init__(self):
        # initialize to appropriate sizes, fill with Gaussian entries
        mean, std = 0.0, 0.01
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

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

## Problem 3, part (d): ReLU Layer
class relu_t:
    def __init__(self):
        pass
    def zero_grad(self):
        pass
    def forward(self, hl):
        # Cache hl in forward because needed for back
        self.hl = hl

        # Compute h_{l + 1}
        hl_plus_1 = np.maximum(0, self.hl)

        return hl_plus_1
    
    def backward(self, dhl_plus_1):
        return np.where(self.hl <= 0, 0, dhl_plus_1)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

## Problem 3, part (e): Combined Softmax + Cross Entropy Loss Layer
class softmax_cross_entropy_t:
    def __init__(self):
        pass

    def zero_grad(self):
        pass

    def forward(self, hl, y):
        # Cache hl in forward because needed for backwards propagation
        self.hl = hl

        # Flatten y as sanity check and store it for usage
        self.y = y.flatten()

        # Step 1: Compute hl_plus_1 after Softmax and Cache it for use in backwards propagation
        exp_hl = np.exp(hl)
        hl_plus_1 = exp_hl / np.sum(exp_hl, axis = 1, keepdims=True)
        self.hl_plus_1 = hl_plus_1
        
        # Step 2: Compute average loss over minibatch 
        B = hl.shape[0]

        # Pick the probabilities corresponding to correct labels and compute Loss
        correct_probs = hl_plus_1[np.arange(B), self.y]
        ell = -np.mean(np.log(correct_probs + 1e-12)) #Adding 1e-12 for numerical stability

        # Step 3: Compute classification error 
        y_pred = np.argmax(hl_plus_1, axis=1)
        error = np.mean(y_pred != self.y)

        return ell, error

    def backward(self):
        B = self.hl.shape[0] # Batch Size

        # Get output from softmax
        softmax_output = self.hl_plus_1

        # Create one hot labels
        y_one_hot = np.zeros_like(softmax_output)
        y_one_hot[np.arange(B), self.y] = 1

        # Compute dhl
        dhl = (softmax_output - y_one_hot) / B
        return dhl

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

## Problem 3, Part (f): Check implementation of forward and backward functionalities for all layers [Embedding, Linear, ReLU, Softmax + Cross Entropy Loss]

### Part 1: Checking Linear Layer Backward Propagation
#indices_W must be passed as a list of tuples, indices_b must be passed as list, and indices_h must be passed as list
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

    # Sample 10 values of k
    k_values = rng.choice(10, size=10, replace=False)

    for k in k_values:
        # For W (10 x 392), pick 10 random (i, j) pairs
        indices_W = [(rng.integers(0, 10), rng.integers(0, 392)) for _ in range(10)]
        
        # For b (10,), pick 10 random indices
        indices_b = rng.choice(10, size=10, replace=False).tolist()
        
        # For h (392,), pick 10 random indices
        indices_h = rng.choice(392, size=10, replace=False).tolist()

        # Run the gradient check for this k
        check_backward_linear(k, indices_W, indices_b, indices_h)

test_backward_linear_random_indices()

### Part 2: Checking ReLU Backward Propagation
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

    # Sample 10 values of k
    k_values = rng.choice(10, size=10, replace=False)

    for k in k_values:
        # For h (10,), pick 10 random indices
        indices_h = rng.choice(10, size=10, replace=False).tolist()

        # Run the gradient check for this k
        check_backward_relu(k, indices_h)

test_backward_relu_random_indices()

### Part 3: Check Backward Propagation of Softmax + Cross Entropy Loss Layer
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

    # For h (10,), pick 10 random indices
    indices_h = rng.choice(10, size=10, replace=False).tolist()

    # Run the gradient check for this k
    check_backward_softmax(indices_h)
    
test_backward_softmax_random_indices()

### Part 4: Check Backward Propagation of Embedding Layer
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

    # Sample 10 values of k
    t_values = rng.choice(10, size=10, replace=False)

    for t in t_values:
        # For W (4 x 4 x 8), pick 10 random (i, j) pairs
        indices_W = [(rng.integers(0, 4), rng.integers(0, 4), rng.integers(0, 8)) for _ in range(10)]
        
        # For b (8,), pick 8 random indices
        indices_b = rng.choice(8, size=8, replace=False).tolist()
        
        # For h (28, 28), pick 10 random indices
        indices_h = [(rng.integers(0, 28), rng.integers(0, 28)) for _ in range(10)]

        # Run the gradient check for this k
        check_backward_embedding(t, indices_W, indices_b, indices_h)

test_backward_embedding_random_indices()

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

## part (g) and (h): Training the Network using SGD. Code for part (g) and (h) is combined because the validate() function 
## from part (h) was used within the training loop for (g) to print out training and validation loss/error

### We start by initializing all Layers
l1, l2, l3, l4 = embedding_t(), linear_t(), relu_t(), softmax_cross_entropy_t()

### Store initial weights: Will be useful for part (i) as I initialize the PyTorch implementation with these exact same weights
### The reason I do this is because I want an accurate comparison between my NumPy implementation and PyTorch's implementation
l1_weights_init, l1_bias_init = l1.w, l1.b
l2_weights_init, l2_bias_init = l2.w, l2.b

### Create Neural Network with the above layers
net = [l1, l2, l3, l4]
print("Part g: Print shapes of X_train, y_train, X_val, and y_val again")
print(f"X_train shape -> {X_train.shape}, y_train shape: {y_train.shape}, X_val shape -> {X_val.shape}, y_val shape: {y_val.shape}")

# X: Full set of samples, y: Full set of labels, B: Batch Size
def validate(l1_w, l1_b, l2_w, l2_b, X, y, B):
    # 1. iterate over mini-batches from the dataset (X, y)
    loss, tot_error = 0, 0
    for i in range(0, X.shape[0], B):
        X_batch, y_batch = X[i: i + B], y[i: i + B]

        # Compute forward pass
        h1 = embedding_utility(X_batch, l1_w, l1_b)
        h2 = h1 @ l2_w.T + l2_b
        h3 = np.maximum(0, h2)
        batch_loss, batch_error = softmax_cross_entropy_utility(h3, y_batch)

        # Accumulate the loss and error
        loss += batch_loss * B
        tot_error += batch_error * B
    
    avg_loss, avg_error = loss / X.shape[0], tot_error/ X.shape[0]
    return avg_loss, avg_error

# Train for at least 1000 iterations with Batch Size 32, learning rate 0.1
B = 32
lr = 0.1
epochs = 10000

# Compute initial training loss/error
training_loss, training_error = validate(l1.w, l1.b, l2.w, l2.b, X_train, y_train, B)

# Compute initial validation loss/error
validation_loss, validation_error = validate(l1.w, l1.b, l2.w, l2.b, X_val, y_val, B)

# Store initial training and validation loss/error
training_losses, training_errors, validation_losses, validation_errors = [training_loss], [training_error], [validation_loss], [validation_error]

# Store loss and error over each batch
batch_losses, batch_errors = [], []

### Store order of batches: Again, this will be useful for part (i) to ensure that PyTorch implementation uses the exact same batches in the exact same order
batch_indices = np.empty((epochs, B), dtype=np.int64)
for i in range(epochs):
    batch_indices[i] = np.random.choice(X_train.shape[0], size=B, replace=False)

### Start training!
for t in range(epochs):
    # 1. sample a mini-batch of size = 32 where each image in the mini-batch is chosen uniformly randomly from the training dataset
    indices = batch_indices[t, :].flatten()
    X_train_batch, y_train_batch = X_train[indices], y_train[indices]

    # 2. zero gradient buffer
    for l in net:
        l.zero_grad()

    # 3. Forward Pass
    h1 = l1.forward(X_train_batch)
    h2 = l2.forward(h1)
    h3 = l3.forward(h2)
    ell, error = l4.forward(h3, y_train_batch)
    batch_losses.append(ell)
    batch_errors.append(error)

    # 4. Backward Pass
    dh3 = l4.backward()
    dh2 = l3.backward(dh3)
    dh1 = l2.backward(dh2)
    dx = l1.backward(dh1)

    # 5. Gather Backprop gradients
    dw1, db1 = l1.dw, l1.db
    dw2, db2 = l2.dw, l2.db
    
    # Store training loss/error every 10 weight updates
    if (t + 1) % 10 == 0:
        training_loss, training_error = validate(l1.w, l1.b, l2.w, l2.b, X_train, y_train, B)
        print(f"Epoch: {t + 1}, Training Loss: {training_loss}, Training Error: {training_error}")
        training_losses.append(training_loss)
        training_errors.append(training_error)
    
    # 6. Store validation loss/error every 10 weight updates
    if (t + 1) % 10 == 0:
        validation_loss, validation_error = validate(l1.w, l1.b, l2.w, l2.b, X_val, y_val, B)
        print(f"Epoch: {t + 1}, Validation Loss: {validation_loss}, Validation Error: {validation_error}")
        validation_losses.append(validation_loss)
        validation_errors.append(validation_error)

    # 7. one step of SGD
    l1.w = l1.w - lr*dw1
    l1.b = l1.b - lr*db1
    l2.w = l2.w - lr*dw2
    l2.b = l2.b - lr*db2

# Store plot of training loss + error
training_epoch_labels = [i * 10 for i in range(len(training_losses))]
validation_epoch_labels = [i * 10 for i in range(len(validation_losses))]
plt.figure(figsize=(8, 5))
plt.plot(training_epoch_labels, training_losses, label='Training Loss[Ravi]')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss vs Iteration')
plt.legend()
plt.grid(True)
plt.savefig('training_loss_plot_ravi.png', dpi=300)

plt.figure(figsize=(8, 5))
plt.plot(training_epoch_labels, training_errors, label='Training Error[Ravi]')
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title('Training Error vs Iteration')
plt.legend()
plt.grid(True)
plt.savefig('training_error_plot_ravi.png', dpi=300)

# Store plot of validation loss + error
plt.figure(figsize=(8, 5))
plt.plot(validation_epoch_labels, validation_losses, label='Validation Loss[Ravi]')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Validation Loss vs Iteration')
plt.legend()
plt.grid(True)
plt.savefig('validation_loss_plot_ravi.png', dpi=300)

plt.figure(figsize=(8, 5))
plt.plot(validation_epoch_labels, validation_errors, label='Validation Error[Ravi]')
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title('Validation Error vs Iteration')
plt.legend()
plt.grid(True)
plt.savefig('validation_error_plot_ravi.png', dpi=300)

## Plot Batch Errors and Batch Losses
plt.figure(figsize=(8, 5))
plt.plot(list(range(1, epochs + 1)), batch_errors, label = "Batch Error[Ravi]")
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title('Batch Error vs Iteration')
plt.legend()
plt.grid(True)
plt.savefig('batch_error_plot_ravi.png', dpi=300)

plt.figure(figsize=(8, 5))
plt.plot(list(range(1, epochs + 1)), batch_losses, label = "Batch Loss[Ravi]")
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Batch Loss vs Iteration')
plt.legend()
plt.grid(True)
plt.savefig('batch_loss_plot_ravi.png', dpi=300)

# Store Final Metrics as a Dictionary
import pandas as pd
final_metrics = {
    "Training Error": [training_errors[-1]],
    "Training Loss": [training_losses[-1]],
    "Validation Error": [validation_errors[-1]],
    "Validation Loss": [validation_losses[-1]]
}

# Convert the above dictionary to DataFrame and save it to a csv file for storage
df = pd.DataFrame(final_metrics)
df.to_csv("final_metrics_ravi.csv", index=False)

### Part (i): Test Implementation using PyTorch. Follows exactly the same steps as NumPy Implementation
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Define class for Embedding Layer
class EmbeddingLayer(nn.Module):
    def __init__(self):
        super().__init__()

        ## 2D Convolution Layer
        self.conv = nn.Conv2d(
            in_channels = 1,
            out_channels = 8,
            kernel_size = 4,
            stride = 4,
            bias = True
        )

        ## Initialize weights and bias to be same as my NumPy implementation
        self._init_weights()
    
    def _init_weights(self):
        with torch.no_grad():
            l1_weights_init_reshaped = l1_weights_init.transpose(2, 0, 1)[:, None, :, :] # PyTorch expects weights to be in the shape (out_channels, in_channels, height, width)
            self.conv.weight.copy_(torch.from_numpy(l1_weights_init_reshaped))
            self.conv.bias.copy_(torch.from_numpy(l1_bias_init))
    
    # x: B X 28 X 28
    def forward(self, x):
        x = x.unsqueeze(1) # Becomes B X 1 X 28 X 28
        x = self.conv(x) # Becomes B X 8 x 7 x 7
        x = x.permute(0, 2, 3, 1) # Becomes B x 7 x 7 x 8
        x = x.flatten(start_dim = 1) # Becomes B X 392
        return x

# Define Overall Neural Network with all the required layers. I note that Softmax + Cross Entropy Loss is implemented in the loss function in PyTorch
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.embed = EmbeddingLayer()
        self.fc = nn.Linear(392, 10)
        self._init_fc()
    
    ## Initialize weights and bias to be same as my NumPy implementation
    def _init_fc(self):
        with torch.no_grad():
            self.fc.weight.copy_(torch.from_numpy(l2_weights_init))
            self.fc.bias.copy_(torch.from_numpy(l2_bias_init))
    
    def forward(self, x):
        x = self.embed(x)
        x = self.fc(x)
        x = F.relu(x)
        return x

## Hyperparameters[Same as NumPy Implementation]
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

# Method to compute loss and error over the entire dataset (used for training + validation within the training loop)
def validate_torch(X, y):
    model.eval()
    tot_loss, tot_error = 0.0, 0.0

    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            # Get batch
            X_batch, y_batch = X[i: i + batch_size], y[i: i + batch_size]

            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # Compute batch error (percentage misclassified)
            _, predicted = torch.max(outputs, 1)
            error = (predicted != y_batch).float().mean().item()

            # Accumulate weighted loss and error
            tot_loss += loss.item() * X_batch.size(0)
            tot_error += error * X_batch.size(0)
    
    # Average over all samples
    avg_loss, avg_error = tot_loss / X.shape[0], tot_error / X.shape[0]
    return avg_loss, avg_error

# Compute initial training loss/error
initial_train_loss, initial_train_error = validate_torch(X_train_tensor, y_train_tensor)

# Compute initial validation loss/error
initial_val_loss, initial_val_error = validate_torch(X_val_tensor, y_val_tensor)

# Initialize the training and validation losses/errors lists
training_losses, training_errors, validation_losses, validation_errors = [initial_train_loss], [initial_train_error], [initial_val_loss], [initial_val_error]

# Store batch losses and batch errors
batch_errors, batch_losses = [], []

# Start Training Loop: Steps are exactly the same as NumPy implementation
for epoch in range(epochs):
    model.train() # Set Model in Training Mode
    
    # Randomly sample a batch
    indices = batch_indices[epoch, :].flatten()
    batch_X, batch_y = X_train_tensor[indices], y_train_tensor[indices]
    
    # Zero out gradients
    optimizer.zero_grad()

    ## Obtain loss and do back propagation
    outputs = model(batch_X)
    loss = criterion(outputs, batch_y)
    loss.backward()
    optimizer.step()

    ## Store batch losses and batch errors
    batch_losses.append(loss.item())
    predictions = torch.argmax(outputs, dim = 1)
    batch_errors.append(torch.mean((predictions != batch_y).float()))
    
    # Store training loss/error every 10 weight updates
    if (epoch + 1) % 10 == 0:
        avg_train_loss, avg_train_error = validate_torch(X_train_tensor, y_train_tensor)
        print(f"Epoch: {epoch + 1}, Training Loss: {avg_train_loss}, Training Error: {avg_train_error}")
        training_losses.append(avg_train_loss)
        training_errors.append(avg_train_error)

    # Store validation loss/error every 10 weight updates
    if (epoch + 1) % 10 == 0:
        avg_val_loss, avg_val_error = validate_torch(X_val_tensor, y_val_tensor)
        print(f"Epoch: {epoch + 1}, Validation Loss: {avg_val_loss}, Validation Error: {avg_val_error}")
        validation_losses.append(avg_val_loss)
        validation_errors.append(avg_val_error)        

# Plot training + validation losses
training_epoch_labels = [i * 10 for i in range(len(training_losses))]
validation_epoch_labels = [i * 10 for i in range(len(validation_losses))]
plt.figure(figsize=(8, 5))
plt.plot(training_epoch_labels, training_losses, label='Training Loss[PyTorch]')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss vs Iteration')
plt.legend()
plt.grid(True)
plt.savefig('training_loss_plot_torch.png', dpi=300)

plt.figure(figsize=(8, 5))
plt.plot(validation_epoch_labels, validation_losses, label='Validation Loss[PyTorch]')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Validation Loss vs Iteration')
plt.legend()
plt.grid(True)
plt.savefig('validation_loss_plot_torch.png', dpi=300)

# Plot training + validation errors
plt.figure(figsize=(8, 5))
plt.plot(training_epoch_labels, training_errors, label='Training Error[PyTorch]')
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title('Training Error vs Iteration')
plt.legend()
plt.grid(True)
plt.savefig('training_error_plot_torch.png', dpi=300)

plt.figure(figsize=(8, 5))
plt.plot(validation_epoch_labels, validation_errors, label='Validation Error[PyTorch]')
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title('Validation Error vs Iteration')
plt.legend()
plt.grid(True)
plt.savefig('validation_error_plot_torch.png', dpi=300)

## Plot Batch Errors and Batch Losses
plt.figure(figsize=(8, 5))
plt.plot(list(range(1, epochs + 1)), batch_errors, label = "Batch Error[PyTorch]")
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title('Batch Error vs Iteration')
plt.legend()
plt.grid(True)
plt.savefig('batch_error_plot_torch.png', dpi=300)

plt.figure(figsize=(8, 5))
plt.plot(list(range(1, epochs + 1)), batch_losses, label = "Batch Loss[PyTorch]")
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Batch Loss vs Iteration')
plt.legend()
plt.grid(True)
plt.savefig('batch_loss_plot_torch.png', dpi=300)

# Store Final Metrics as a Dictionary
final_metrics = {
    "Training Error": [training_errors[-1]],
    "Training Loss": [training_losses[-1]],
    "Validation Error": [validation_errors[-1]],
    "Validation Loss": [validation_losses[-1]]
}

# Convert the above dictionary to DataFrame and save it to a csv file for storage
df = pd.DataFrame(final_metrics)
df.to_csv("final_metrics_torch.csv", index=False)
